from easydict import EasyDict
import torch

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