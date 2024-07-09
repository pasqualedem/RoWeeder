import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FV

from selfweed.data.utils import DataDict, crop_to_nonzero
from selfweed.detector import HoughDetectorDict
from selfweed.labeling import get_drawn_img, get_slic, label_from_row
from selfweed.models.utils import ModelOutput

class HoughSLICSegmentationWrapper(nn.Module):
    classificator_size = (224, 224)
    def __init__(self, classification_model, plant_detector, slic_params, use_ndvi=True) -> None:
        super().__init__()
        self.model = classification_model
        self.plant_detector = plant_detector
        self.slic_params = slic_params
        self.use_ndvi = use_ndvi
        self.__repr__ = f"HoughSlicWrapper:\n{self.model.__repr__}"
        
    def segment(self, image, mask, slic):
        weedmap = mask.clone().long()
        slic_mask = slic * mask if self.use_ndvi else slic
        unique_slic = slic_mask.unique()
        for u in unique_slic[1:]:
            patch = slic_mask == u
            img_masked = image * patch
            img_patch = crop_to_nonzero(img_masked)
            img_patch = FV.resize(img_patch, self.classificator_size).unsqueeze(0)
            logits = self.model(img_patch).logits
            label = logits.argmax(dim=1) + 1
            weedmap[patch] = label
        weedmap = F.one_hot(weedmap, num_classes=3).permute(0, 3, 1, 2).float()
        return weedmap
        
    def forward(self, image, ndvi=None):
        if self.training or ndvi is None:
            return self.model(image)
        B, _, H, W = image.shape
        segmentations = []
        for i in range(image.shape[0]):
            mask = self.plant_detector(ndvi=ndvi[i])[0]
            slic = torch.tensor(get_slic(image[i], self.slic_params), device=mask.device)
            segmentations.append(self.segment(image[i], mask, slic))
        return ModelOutput(logits=torch.cat(segmentations), scores=None)

    def get_learnable_params(self, train_params):
        return self.model.get_learnable_params(train_params)