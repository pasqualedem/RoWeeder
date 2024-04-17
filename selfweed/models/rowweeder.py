import enum
import torch
import torch.nn as nn

from torch.nn.functional import normalize
from einops.layers.torch import Reduce
from einops import rearrange, repeat

from selfweed.models.utils import RowWeederModelOutput
    

class RowWeeder(nn.Module):
    def __init__(self, encoder, input_channels, embedding_dim, transformer_layers=4) -> None:
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        num_channels = len(input_channels)
        self.plant_encoder = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Reduce("b c h w -> b c", reduction="mean"),
        )
        self.no_crop_embedding = nn.Embedding(1, embedding_dim)
        self.no_weed_embedding = nn.Embedding(1, embedding_dim)
        
        self.crop_embedding = nn.Embedding(1, embedding_dim)
        self.weed_embedding = nn.Embedding(1, embedding_dim)
        self.transformer = CropWeedTransformer(embedding_dim, num_layers=transformer_layers)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
        )
        
    def _encode_plants(self, plants):
        plants, flags = plants
        b = plants.shape[0]
        if flags.sum() == 0:
            return None
        plants = rearrange(plants, "b n c h w -> (b n) c h w")
        plants_features = self.plant_encoder(plants)
        return rearrange(plants_features, "(b n) c -> b n c", b=b)
        
    def _get_scores(self, crop_features, weed_features):
        if crop_features is not None:
            normalized_crop_features = normalize(crop_features, p=2, dim=-1)
            normalized_crops_embeddings = normalize(self.crop_embedding.weight, p=2, dim=-1)
            crop_scores = normalized_crop_features @ normalized_crops_embeddings
            neg_crop_scores = normalized_crop_features @ normalized_weed_embeddings
        else:
            crop_scores = torch.zeros(1, 1, 1, 1)
            neg_crop_scores = torch.zeros(1, 1, 1, 1)
        
        if weed_features is not None:
            normalized_weed_features = normalize(weed_features, p=2, dim=-1)
            normalized_weed_embeddings = normalize(self.weed_embedding.weight, p=2, dim=-1)
            weed_scores = normalized_weed_features @ normalized_weed_embeddings
            neg_weed_scores = normalized_weed_features @ normalized_crops_embeddings
        else:
            weed_scores = torch.zeros(1, 1, 1, 1)
            neg_weed_scores = torch.zeros(1, 1, 1, 1)
        
        score_matrix = torch.stack(
            [crop_scores, weed_scores, neg_crop_scores, neg_weed_scores], dim=1
        )
        return rearrange(score_matrix, "b (r c) d -> b r c d", r=2, c=2)
    
    def get_learnable_params(self, train_params):
        if train_params.get("freeze_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False
        return filter(lambda p: p.requires_grad, self.parameters())
        
    def forward(self, image, crops, weeds):
        features = self.encoder(image)  # (B, C, H, W)

        crop_features = self._encode_plants(crops)  # (B x N, C)
        weed_features = self._encode_plants(weeds)  # (B x N, C)
        score_matrix = self._get_scores(features, crop_features, weed_features)
        
        cropweed_embedding = torch.cat(
            [self.crop_embedding.weight, self.weed_embedding.weight], dim=0)
        features = self.transformer(features, cropweed_embedding)
        logits = self.decoder(features)
        return RowWeederModelOutput(logits=logits, scores=score_matrix)


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
        features = rearrange(features, "b c h w -> b c (h w)")
        features = self.crop_weed_attention(features, cropweed_embedding, cropweed_embedding)
        features =  self.norm(features)
        features = self.mlp(features)
        return rearrange(features, "b c (h w) -> b c h w", h=H, w=W)
        

class CropWeedTransformer(nn.Module):
    def __init__(self, embedding_dim, num_layers, num_heads=8, dropout=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.layers = nn.ModuleList([
            CropWeedAttention(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, features, cropweed_embedding):
        for layer in self.layers:
            features = layer(features, cropweed_embedding)
        return features