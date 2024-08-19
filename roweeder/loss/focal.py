import torch
import torch.nn.functional as F
from torch.nn import Module
from .utils import get_reduction


class FocalLoss(Module):
    def __init__(
        self, gamma: float = 2.0, reduction: str = "mean", **kwargs
    ):
        super().__init__()
        self.gamma = gamma

        self.reduction = get_reduction(reduction)

    def __call__(self, x, target, weight_matrix=None, **kwargs):
        ce_loss = F.cross_entropy(x, target, reduction="none")
        pt = torch.exp(-ce_loss)
        if weight_matrix is not None:
            focal_loss = torch.pow((1 - pt), self.gamma) * weight_matrix * ce_loss
        else:
            focal_loss = torch.pow((1 - pt), self.gamma) * ce_loss

        return self.reduction(focal_loss)
    
    
class PlantLoss(FocalLoss):

    def __call__(self, x, target, **kwargs):
        # Merge class 1 and 2
        target = target.where(target != 2, 1)
        plant_logits = x[:, 1:3].mean(dim=1)
        logits = torch.cat((x[:, 0].unsqueeze(1), plant_logits.unsqueeze(1)), dim=1)
        return super().__call__(logits, target, **kwargs)