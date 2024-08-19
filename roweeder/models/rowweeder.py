import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import normalize
from einops.layers.torch import Reduce
from einops import rearrange, repeat

from roweeder.models.utils import RowWeederModelOutput


class RowWeeder(nn.Module):
    def __init__(
        self,
        encoder,
        input_channels,
        embedding_dims,
        embedding_size=(1,),
        transformer_layers=4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.embedding_dims = embedding_dims
        self.embedding_size = embedding_size
        self.min_plant_size = 32
        num_channels = len(input_channels)
        effective_embedding_dim = [embedding_dim * math.prod(embedding_size) for embedding_dim in embedding_dims]
        channels = [len(input_channels)] + embedding_dims
        self.plant_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                for i in range(len(channels) - 1)
            ]
        )
        self.crop_embeddings = nn.ModuleList(
            [
                nn.Embedding(1, embedding_dim)
                for embedding_dim in effective_embedding_dim
            ]
        )
        self.weed_embeddings = nn.ModuleList(
            [
                nn.Embedding(1, embedding_dim)
                for embedding_dim in effective_embedding_dim
            ]
        )
        self.transformers = nn.ModuleList(
            [
                CropWeedTransformer(embedding_dim, transformer_layers)
                for embedding_dim in effective_embedding_dim
            ]
        )
        decoder_layers = []
        reverse_channels = effective_embedding_dim[::-1]
        for i in range(num_channels):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        reverse_channels[i],
                        reverse_channels[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.GELU(),
                )
            )
        decoder_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    reverse_channels[-1],
                    1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
        )
        self.decoder = nn.ModuleList(decoder_layers)

    def _encode_plants(self, plants):
        if plants is None:
            return None
        plants, flags = plants
        b = plants.shape[0]
        if flags.sum() == 0:
            return None
        plants = rearrange(plants, "b n c h w -> (b n) c h w")

        # Ensure min plant size
        H, W = plants.shape[-2:]
        if H < self.min_plant_size or W < self.min_plant_size:
            if H < W:
                H_W_ratio = W / H
                W = int(self.min_plant_size * H_W_ratio)
                plants = F.interpolate(
                    plants, size=(self.min_plant_size, W), mode="bilinear"
                )
            else:
                W_H_ratio = H / W
                H = int(self.min_plant_size * W_H_ratio)
                plants = F.interpolate(
                    plants, size=(H, self.min_plant_size), mode="bilinear"
                )

        plants_pyramid_features = []
        plants_features = plants
        for encoder in self.plant_encoder:
            plants_features = encoder(plants_features)
            if len(self.embedding_size) > 1:
                plants_embedding = F.adaptive_avg_pool2d(
                    plants_features, self.embedding_size[:2]
                )
                plants_embedding = rearrange(
                    plants_embedding, "(b n) h w c -> (b n) (h w c)", b=b
                )
            else:
                plants_embedding = plants_features.mean(dim=(2, 3))
            plants_embedding = rearrange(plants_embedding, "(b n) c -> b n c", b=b)
            plants_pyramid_features.append(plants_embedding)
        return plants_pyramid_features

    def _get_scores(self, crop_features, weed_features, i, batch_size, device):
        normalized_crops_embeddings = normalize(
            self.crop_embeddings[i].weight, p=2, dim=-1
        )
        normalized_weed_embeddings = normalize(
            self.weed_embeddings[i].weight, p=2, dim=-1
        )
        if crop_features is not None:
            normalized_crop_features = normalize(crop_features[i], p=2, dim=-1)
            normalized_crop_features = rearrange(
                normalized_crop_features, "b n c -> (b n) c"
            )
            crop_scores = normalized_crop_features @ normalized_crops_embeddings.T
            neg_crop_scores = normalized_crop_features @ normalized_weed_embeddings.T
            crop_scores = torch.stack([crop_scores, neg_crop_scores], dim=1)
        else:
            crop_scores = torch.ones(batch_size, 2, 0, device=device)

        if weed_features is not None:
            normalized_weed_features = normalize(weed_features[i], p=2, dim=-1)
            normalized_weed_features = rearrange(
                normalized_weed_features, "b n c -> (b n) c"
            )
            weed_scores = normalized_weed_features @ normalized_weed_embeddings.T
            neg_weed_scores = normalized_weed_features @ normalized_crops_embeddings.T
            weed_scores = torch.stack([weed_scores, neg_weed_scores], dim=1)
        else:
            weed_scores = torch.ones(batch_size, 2, 0, device=device)

        return [crop_scores, weed_scores]

    def _decode_background(self, features):
        features = features[::-1]
        for i in range(len(features) - 1):
            features[i] = self.decoder[i](features[i])
            features[i + 1] += features[i]
        return self.decoder[-1](features[-1])

    def get_learnable_params(self, train_params):
        if train_params.get("freeze_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False
        params = filter(lambda p: p.requires_grad, self.parameters())
        return [{"params": params}]

    def forward(self, image, crops=None, weeds=None):
        B, _, H, W = image.shape
        features = self.encoder(
            image, output_hidden_states=True
        ).hidden_states  # (B, C, H, W) for each layer

        crop_features = self._encode_plants(crops)  # (B x N, C) for each layer
        weed_features = self._encode_plants(weeds)  # (B x N, C) for each layer
        contrastive_scores = [
            self._get_scores(crop_features, weed_features, i, B, image.device)
            for i in range(len(self.embedding_dims))
        ]

        cropweed_embeddings = [
            torch.cat(
                [self.crop_embeddings[i].weight, self.weed_embeddings[i].weight], dim=0
            )
            for i in range(len(self.embedding_dims))
        ]
        cropweed_embeddings = [
            repeat(cropweed_embeddings[i], "c e -> b c e", b=image.shape[0])
            for i in range(len(self.embedding_dims))
        ]
        if len(self.embedding_size) > 1:
            features = [
                rearrange(
                    F.unfold(
                        feature,
                        self.embedding_size,
                        padding=(emb_size // 2 for emb_size in self.embedding_size),
                    ),
                    "b c (h w) -> b c h w",
                    h=feature.shape[2],
                )
                for feature in features
            ]
        features = [
            self.transformers[i](features[i], cropweed_embeddings[i])
            for i in range(len(self.embedding_dims))
        ]
        H4, W4 = features[0].shape[-2:]
        logits = []
        for feature, cropweed_embedding in zip(features, cropweed_embeddings):
            feature = F.interpolate(feature, size=(H4, W4), mode="bilinear")
            cropweed_embedding = rearrange(cropweed_embedding, "b e c -> b c e")
            feature = rearrange(feature, "b c h w -> b (h w) c")
            feature = feature @ cropweed_embedding
            feature = rearrange(feature, "b (h w) c -> b c h w", h=H4, w=W4)
            logits.append(feature)
        logits = torch.stack(logits, dim=1).mean(dim=1)
        background_logits = self._decode_background(features)
        logits = torch.cat([logits, background_logits], dim=1)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear")
        return RowWeederModelOutput(logits=logits, scores=contrastive_scores)


class CropWeedAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.crop_weed_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(self, features, cropweed_embedding):
        B, C, H, W = features.shape
        features = rearrange(features, "b c h w -> (h w) b c")
        cropweed_embedding = rearrange(cropweed_embedding, "b e c -> e b c")
        features, _ = self.crop_weed_attention(
            features, cropweed_embedding, cropweed_embedding
        )
        features = self.norm(features)
        features = self.mlp(features)
        return rearrange(features, "(h w) b c -> b c h w", h=H, w=W)


class CropWeedTransformer(nn.Module):
    def __init__(self, embedding_dim, num_layers, num_heads=8, dropout=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.layers = nn.ModuleList(
            [
                CropWeedAttention(embedding_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, features, cropweed_embedding):
        for layer in self.layers:
            features = layer(features, cropweed_embedding)
        return features
