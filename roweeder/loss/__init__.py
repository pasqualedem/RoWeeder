import torch.nn as nn
import torch

from roweeder.models.utils import LossOutput, RowWeederModelOutput

from .focal import FocalLoss, PlantLoss
from .contrastive import ContrastiveLoss
from .utils import get_weight_matrix_from_labels


LOGITS_LOSSES = {"focal": FocalLoss, "plant": PlantLoss}

EMBEDDING_LOSSES = {
    "contrastive": ContrastiveLoss,
}


class RowLoss(nn.Module):
    """This loss is a linear combination of the following losses:
    - FocalLoss
    - ContrastiveLoss
    """

    def __init__(self, components, class_weighting=None):
        super().__init__()
        self.weights = {k: v.pop("weight") for k, v in components.items()}
        self.logits_components = nn.ModuleDict(
            [
                [k, LOGITS_LOSSES[k](**v)]
                for k, v in components.items()
                if k in LOGITS_LOSSES
            ]
        )
        self.embedding_components = nn.ModuleDict(
            [
                [k, EMBEDDING_LOSSES[k](**v)]
                for k, v in components.items()
                if k in EMBEDDING_LOSSES
            ]
        )
        if (
            set(components.keys())
            - set(self.logits_components.keys())
            - set(self.embedding_components.keys())
        ):
            raise ValueError(
                f"Unknown loss components: {set(components.keys()) - set(self.logits_components.keys())}"
            )
        self.class_weighting = class_weighting
        self.components = {**self.logits_components, **self.embedding_components}

    def logits_loss(self, logits, target):
        weight_matrix, class_weights = None, None
        if self.class_weighting:
            num_classes = logits.shape[1]
            weight_matrix, class_weights = get_weight_matrix_from_labels(
                target, num_classes
            )
        return {
            k: self.weights[k]
            * loss(
                logits, target, weight_matrix=weight_matrix, class_weights=class_weights
            )
            for k, loss in self.logits_components.items()
        }

    def contrastive_loss(self, result: RowWeederModelOutput):
        return {
            k: self.weights[k] * loss(result.scores)
            for k, loss in self.embedding_components.items()
        }

    def forward(self, result: RowWeederModelOutput, target):
        if isinstance(result, torch.Tensor):  # Only logits
            logits_loss = self.logits_loss(result, target)
            return logits_loss

        logits_loss_dict = self.logits_loss(result.logits, target)
        prompt_loss_dict = self.contrastive_loss(result)
        loss_value = sum(logits_loss_dict.values()) + sum(prompt_loss_dict.values())
        return LossOutput(
            value=loss_value, components=dict(**logits_loss_dict, **prompt_loss_dict)
        )


def build_loss(params):
    return RowLoss(**params)
