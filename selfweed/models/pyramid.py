import torch
import torch.nn as nn
import torch.nn.functional as F

from selfweed.models.utils import RowWeederModelOutput

class PyramidFormer(nn.Module):
    def __init__(
        self,
        encoder,
        embedding_dims,
        num_classes,
        fusion="concat",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.embedding_dims = embedding_dims
        
        self.pyramid_fusers = nn.ModuleList([
            PyramidFuser(dim_shallow, dim_deep, fusion=fusion) for dim_shallow, dim_deep in zip(embedding_dims[::-1], embedding_dims[::-1][1:])
        ])
        self.classifier = nn.Conv2d(embedding_dims[0], num_classes, kernel_size=1)
        
    def forward(self, image):
        B, _, H, W = image.shape
        hidden_states = self.encoder(image, output_hidden_states=True).hidden_states
        x = hidden_states[-1]
        for i, fuser in enumerate(self.pyramid_fusers):
            x = fuser(hidden_states[-i-2], x)
        logits = self.classifier(x)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear")
        return RowWeederModelOutput(logits=logits, scores=None)
    
    def get_learnable_params(self, train_params):
        if train_params.get("freeze_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False
        params = filter(lambda p: p.requires_grad, self.parameters())
        return [{"params": params}]
        
        
class PyramidFuser(nn.Module):
    def __init__(self, dim_deep, dim_shallow, activation="GELU", fusion="concat"):
        super().__init__()
        self.fusion = fusion
        if fusion == "concat":
            dim_input = dim_shallow + dim_deep
        elif fusion == "add":
            dim_input = dim_deep 
        self.fuse_conv = nn.Conv2d(dim_input, dim_shallow, kernel_size=1)
        self.spatial_fuse = nn.Conv2d(dim_shallow, dim_shallow, kernel_size=3, padding=1)
        self.activation = getattr(nn, activation)()
        
    def forward(self, x_shallow, x_deep):
        x_deep = F.interpolate(x_deep, size=x_shallow.shape[-2:], mode="bilinear", align_corners=False)
        if self.fusion == "concat":
            x = torch.cat([x_shallow, x_deep], dim=1)
            x = self.fuse_conv(x)
        elif self.fusion == "add":
            x_deep = self.fuse_conv(x_deep)
            x = x_shallow + x_deep
        x = self.spatial_fuse(x)
        x = self.activation(x)
        return x
    
class MLFormer(nn.Module):
    def __init__(
        self,
        encoder,
        embedding_dims,
        num_classes,
        fusion="concat",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.embedding_dims = embedding_dims
        
        self.mlf_fuser = MLFuser(embedding_dims, fusion=fusion)
        self.classifier = nn.Conv2d(embedding_dims[0], num_classes, kernel_size=1)
        
    def forward(self, image):
        B, _, H, W = image.shape
        hidden_states = self.encoder(image, output_hidden_states=True).hidden_states
        x = self.mlf_fuser(hidden_states)
        logits = self.classifier(x)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear")
        return RowWeederModelOutput(logits=logits, scores=None)
    
    def get_learnable_params(self, train_params):
        if train_params.get("freeze_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False
        params = filter(lambda p: p.requires_grad, self.parameters())
        return [{"params": params}]
    
class MLFuser(nn.Module):
    def __init__(self, embedding_dims, activation="GELU", fusion="concat"):
        super().__init__()
        self.fusion = fusion
        if fusion == "concat":
            self.fuse_conv = nn.Conv2d(sum(embedding_dims), embedding_dims[0], kernel_size=1)
        elif fusion == "add":
            self.fuse_conv = nn.ModuleList([
                nn.Conv2d(dim, embedding_dims[0], kernel_size=1) for dim in embedding_dims
            ])
        self.spatial_fuse = nn.Conv2d(embedding_dims[0], embedding_dims[0], kernel_size=3, padding=1)
        self.activation = getattr(nn, activation)()
        
    def forward(self, x):
        x = [
            F.interpolate(x_i, size=x[0].shape[-2:], mode="bilinear", align_corners=False) for x_i in x
        ]
        if self.fusion == "concat":
            x = torch.cat(x, dim=1)
            x = self.fuse_conv(x)
        elif self.fusion == "add":
            x = sum([conv(x_i) for conv, x_i in zip(self.fuse_conv, x)])
        x = self.spatial_fuse(x)
        x = self.activation(x)
        return x