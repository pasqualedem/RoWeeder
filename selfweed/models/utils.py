import torch
import torch.nn as nn
import torch.nn.functional as F

from selfweed.utils.utils import EasyDict

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