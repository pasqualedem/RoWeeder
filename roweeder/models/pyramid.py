import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import PyTorchModelHubMixin
from einops import rearrange

from roweeder.data.weedmap import WeedMapDataset
from roweeder.models.hub import has_config
from roweeder.models.utils import RowWeederModelOutput, get_segformer_encoder


class RoWeederPyramid(nn.Module, PyTorchModelHubMixin):
    @has_config
    def __init__(
        self,
        encoder,
        num_classes,
        embedding_dims=None,
        fusion="concat",
        upsampling="interpolate",
        spatial_conv=True,
        blocks=4,
    ) -> None:
        super().__init__()
        if isinstance(encoder, str):
            encoder, embedding_dims = get_segformer_encoder(encoder)
        self.encoder = encoder
        self.embedding_dims = embedding_dims
        self.blocks = blocks

        self.pyramid_fusers = nn.ModuleList(
            [
                PyramidFuser(dim_shallow, dim_deep, fusion=fusion, upsampling=upsampling, spatial_conv=spatial_conv)
                for dim_shallow, dim_deep in zip(
                    embedding_dims[::-1], embedding_dims[::-1][1:]
                )
            ]
        )
        self.classifier = nn.Conv2d(embedding_dims[0], num_classes, kernel_size=1)

    def forward(self, image):
        B, _, H, W = image.shape
        hidden_states = self.encoder(image, output_hidden_states=True).hidden_states
        if len(hidden_states) > self.blocks:
            hidden_states = hidden_states[:self.blocks]
        x = hidden_states[-1]
        for i, fuser in enumerate(self.pyramid_fusers):
            x = fuser(hidden_states[-i - 2], x)
        logits = self.classifier(x)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear")
        return RowWeederModelOutput(logits=logits, scores=None)

    def get_learnable_params(self, train_params):
        if train_params.get("freeze_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False
        params = filter(lambda p: p.requires_grad, self.parameters())
        return [{"params": params}]
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        encoder, embedding_dims = get_segformer_encoder()
        kwargs["encoder"] = encoder
        kwargs["embedding_dims"] = embedding_dims
        if "num_classes" not in kwargs:
            kwargs["num_classes"] = len(WeedMapDataset.id2class)
            
        return super().from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)

class PyramidFuser(nn.Module):
    def __init__(self, dim_deep, dim_shallow, activation="GELU", fusion="concat", upsampling="interpolate", spatial_conv=True):
        super().__init__()
        self.fusion = fusion
        if fusion == "concat":
            dim_input = dim_shallow + dim_deep
        elif fusion == "add":
            dim_input = dim_deep
        self.fuse_conv = nn.Conv2d(dim_input, dim_shallow, kernel_size=1)
        self.spatial_fuse = nn.Conv2d(
            dim_shallow, dim_shallow, kernel_size=3, padding=1
        ) if spatial_conv else nn.Identity()
        self.activation = getattr(nn, activation)()
        if upsampling == "interpolate":
            self.upsample = lambda x: F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )
        elif upsampling == "deconv":
            self.upsample = nn.ConvTranspose2d(dim_deep, dim_deep, kernel_size=4, stride=2, padding=1)

    def forward(self, x_shallow, x_deep):
        x_deep = self.upsample(x_deep)
        if self.fusion == "concat":
            x = torch.cat([x_shallow, x_deep], dim=1)
            x = self.fuse_conv(x)
        elif self.fusion == "add":
            x_deep = self.fuse_conv(x_deep)
            x = x_shallow + x_deep
        x = self.spatial_fuse(x)
        x = self.activation(x)
        return x


class RoWeederFlat(nn.Module, PyTorchModelHubMixin):
    
    @has_config
    def __init__(
        self,
        encoder,
        num_classes,
        embedding_dims=None,
        fusion="concat",
        upsampling="interpolate",
        spatial_conv=True,
        scale_factors=[2, 4, 8],
    ) -> None:
        super().__init__()
        if isinstance(encoder, str):
            encoder, embedding_dims = get_segformer_encoder(encoder)
        self.encoder = encoder
        self.embedding_dims = embedding_dims

        self.mlf_fuser = MLFuser(embedding_dims, fusion=fusion, upsampling=upsampling, spatial_conv=spatial_conv, scale_factors=scale_factors)
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
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        encoder, embedding_dims = get_segformer_encoder()
        kwargs["encoder"] = encoder
        kwargs["embedding_dims"] = embedding_dims
        if "num_classes" not in kwargs:
            kwargs["num_classes"] = len(WeedMapDataset.id2class)
            
        return super().from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)


class MLFuser(nn.Module):
    def __init__(
        self,
        embedding_dims,
        activation="GELU",
        fusion="concat",
        upsampling="interpolate",
        spatial_conv=True,
        scale_factors=[2, 4, 8],
    ):
        super().__init__()
        self.fusion = fusion
        if fusion == "concat":
            self.fuse_conv = nn.Conv2d(
                sum(embedding_dims), embedding_dims[0], kernel_size=1
            )
        elif fusion == "add":
            self.fuse_conv = nn.ModuleList(
                [
                    nn.Conv2d(dim, embedding_dims[0], kernel_size=1)
                    for dim in embedding_dims
                ]
            )
        if spatial_conv:
            self.spatial_fuse = nn.Conv2d(
                embedding_dims[0], embedding_dims[0], kernel_size=3, padding=1
            )
        else:
            self.spatial_fuse = nn.Identity()
        self.activation = getattr(nn, activation)()
        if upsampling == "interpolate":
            self.upsample = [
                lambda x, scale=scale: F.interpolate(
                    x, scale_factor=scale, mode="bilinear", align_corners=False
                )
                for scale in scale_factors
            ]
        elif upsampling == "deconv":
            self.upsample = nn.ModuleList(
                [
                    nn.Sequential(
                        *[nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)
                        for _ in range(i + 1)]
                    )
                    for i, dim in enumerate(embedding_dims[1:])
                ]
            )
        else:
            raise ValueError(f"Unknown upsampling method {upsampling}")

    def forward(self, x):
        if len(x[0].shape) == 3: # Have to reshape
            shapes = [int(math.sqrt(x_i.shape[1])) for x_i in x]
            x = [rearrange(x_i, "b (h w) c -> b c h w", h=shape, w=shape) for x_i, shape in zip(x, shapes)]

        x = [x[0]] + [upsample(x_i) for upsample, x_i in zip(self.upsample, x[1:])]
        if self.fusion == "concat":
            x = torch.cat(x, dim=1)
            x = self.fuse_conv(x)
        elif self.fusion == "add":
            x = sum(conv(x_i) for conv, x_i in zip(self.fuse_conv, x))
        x = self.spatial_fuse(x)
        x = self.activation(x)
        return x
