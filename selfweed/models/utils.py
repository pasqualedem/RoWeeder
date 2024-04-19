from collections import OrderedDict
from dataclasses import dataclass
import torch

@dataclass
class RowWeederModelOutput(OrderedDict):
    logits: torch.Tensor
    scores: torch.Tensor
    

@dataclass
class ModelOutput(OrderedDict):
    loss: torch.Tensor
    logits: torch.Tensor
    scores: torch.Tensor