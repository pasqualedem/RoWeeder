from easydict import EasyDict
import torch

class RowWeederModelOutput(EasyDict):
    logits: torch.Tensor
    scores: torch.Tensor
    

class ModelOutput(EasyDict):
    loss: torch.Tensor
    logits: torch.Tensor
    scores: torch.Tensor
    
class LossOutput(EasyDict):
    value: torch.Tensor
    components: dict