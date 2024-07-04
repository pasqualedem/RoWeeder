import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidFormer(nn.Module):
    def __init__(
        self,
        encoder,
        embedding_dims,
        num_classes,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.embedding_dims = embedding_dims
        
        self.pyramid_fusers = nn.ModuleList([
            PyramidFuser(dim_shallow, dim_deep) for dim_shallow, dim_deep in zip(embedding_dims, embedding_dims[1:])
        ])
        self.classifier = nn.Conv2d(embedding_dims[0], num_classes, kernel_size=1)
        
    def forward(self, x):
        hidden_states = self.encoder(x)
        x = hidden_states[-1]
        for fuser in self.pyramid_fusers:
            x = fuser(hidden_states.pop(-2), x)
        x = self.classifier(x)
        return x
        
        
class PyramidFuser(nn.Module):
    def __init__(self, dim_shallow, dim_deep, activation="GELU"):
        super().__init__()
        self.fuse_conv = nn.Conv2d(dim_shallow + dim_deep, dim_shallow, kernel_size=1)
        self.spatial_fuse = nn.Conv2d(dim_shallow, dim_shallow, kernel_size=3, padding=1)
        self.activation = getattr(nn, activation)()
        
    def forward(self, x_shallow, x_deep):
        x_deep = F.interpolate(x_deep, size=x_shallow.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x_shallow, x_deep], dim=1)
        x = self.fuse_conv(x)
        x = self.spatial_fuse(x)
        x = self.activation(x)
        return x