import os
import torch
import torchvision
from roweeder.data.weedmap import WeedMapDataset
from roweeder.models.utils import ModelOutput


class PseudoModel(torch.nn.Module):
    def __init__(self, gt_folder: str):
        super().__init__()
        self.gt_folder = gt_folder
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        
    def _get_device(self):
        return next(self.parameters()).device

    def forward(self, image, name):
        pseudo_gts = []
        for n in name:
            img_name = os.path.basename(n)
            field = os.path.basename(os.path.dirname(os.path.dirname(n)))
            img_path = os.path.join(self.gt_folder, field, img_name)
            gt = torchvision.io.read_image(img_path)
            gt = gt[[2, 1, 0], ::].bool().float()
            pseudo_gts.append(gt)
        return ModelOutput(logits=torch.stack(pseudo_gts).to(self._get_device()), scores=None)
