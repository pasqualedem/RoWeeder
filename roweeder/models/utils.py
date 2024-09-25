from safetensors import safe_open
from safetensors.torch import save_file

import torch
import torch.nn as nn
import torch.nn.functional as F

from roweeder.utils.utils import EasyDict


class RowWeederModelOutput(EasyDict):
    logits: torch.Tensor
    scores: torch.Tensor


class LossOutput(EasyDict):
    value: torch.Tensor
    components: dict


class ModelOutput(EasyDict):
    loss: LossOutput
    logits: torch.Tensor
    scores: torch.Tensor


class HuggingFaceWrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.__repr__ = self.model.__repr__

    def forward(self, image):
        B, _, H, W = image.shape
        hug_dict = {"pixel_values": image}
        logits = self.model(**hug_dict).logits
        logits = F.interpolate(logits, size=(H, W), mode="bilinear")
        return RowWeederModelOutput(logits=logits, scores=None)

    def get_learnable_params(self, train_params):
        return self.model.parameters()


class HuggingFaceClassificationWrapper(HuggingFaceWrapper):
    def forward(self, image):
        B, _, H, W = image.shape
        hug_dict = {"pixel_values": image}
        logits = self.model(**hug_dict).logits
        return RowWeederModelOutput(logits=logits, scores=None)


def torch_dict_load(file_path):
    if (
        file_path.endswith(".pth")
        or file_path.endswith(".pt")
        or file_path.endswith(".bin")
    ):
        return torch.load(file_path)
    if file_path.endswith(".safetensors"):
        with safe_open(file_path, framework="pt") as f:
            d = {}
            for k in f.keys():
                d[k] = f.get_tensor(k)
        return d
    raise ValueError("File extension not supported")


def torch_dict_save(data, file_path):
    if (
        file_path.endswith(".pth")
        or file_path.endswith(".pt")
        or file_path.endswith(".bin")
    ):
        torch.save(data, file_path)
    elif file_path.endswith(".safetensors"):
        save_file(data, file_path)
    else:
        raise ValueError("File extension not supported")


def load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        print("Could not load state dict, trying to remove the model prefix")
        prefix = "model."
        state_dict = {
            k[len(prefix) :] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)


def get_segformer_encoder(version="nvidia/mit-b0"):
    from transformers import SegformerForImageClassification, SegformerConfig

    encoder = SegformerForImageClassification.from_pretrained(version)
    embeddings_dims = SegformerConfig.from_pretrained(version).hidden_sizes
    return encoder, embeddings_dims