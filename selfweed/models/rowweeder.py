import enum
import torch
import torch.nn as nn

from einops.layers.torch import Reduce
from einops import rearrange, repeat

class ResultDict(enum):
    LOGITS = 0
    SCORES = 1


class RowWeeder(nn.Module):
    def __init__(self, encoder, input_channels, embedding_dim, transformer_layers=4) -> None:
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.plant_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Reduce("b c h w -> b c", reduction="mean"),
        )
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

    def forward(self, image, crops, weeds):
        features = self.encoder(image)  # (B, C, H, W)

        crop_features = self.plant_encoder(crops)  # (B x N, C)
        weed_features = self.plant_encoder(weeds)  # (B x N, C)

        crop_scores = torch.einsum(
            "bc,bc->b", crop_features, self.crop_embedding.weight
        )
        weed_scores = torch.einsum(
            "bc,bc->b", weed_features, self.weed_embedding.weight
        )
        neg_crop_scores = torch.einsum(
            "bc,bc->b", crop_features, self.weed_embedding.weight
        )
        neg_weed_scores = torch.einsum(
            "bc,bc->b", weed_features, self.crop_embedding.weight
        )
        score_matrix = torch.stack(
            [crop_scores, weed_scores, neg_crop_scores, neg_weed_scores], dim=1
        )
        score_matrix = rearrange(score_matrix, "b (r c) d -> b r c d", r=2, c=2)
        
        cropweed_embedding = torch.cat(
            [self.crop_embedding.weight, self.weed_embedding.weight], dim=0)
        features = self.transformer(features, cropweed_embedding)
        logits = self.decoder(features)
        return {
            ResultDict.LOGITS: logits,
            ResultDict.SCORES: score_matrix,
        }


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