import torch
from transformers.utils import ModelOutput


class RowWeederModelOutput(ModelOutput):
    logits: torch.Tensor
    scores: torch.Tensor