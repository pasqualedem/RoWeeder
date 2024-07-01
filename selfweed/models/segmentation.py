import numpy as np
import torch
import torch.nn as nn

from selfweed.data.utils import DataDict, crop_to_nonzero
from selfweed.detector import HoughDetectorDict
from selfweed.labeling import get_drawn_img, get_slic, label_from_row

class HoughSLICSegmentationWrapper(nn.Module):
    def __init__(self, classification_model, plant_detector, slic_params) -> None:
        super().__init__()
        self.model = classification_model
        self.plant_detector = plant_detector
        self.slic_params = slic_params
        self.__repr__ = f"HoughSlicWrapper:\n{self.model.__repr__}"
        
    def segment(self, image, mask, slic):
        weedmap = mask.clone()
        slic_mask = slic * mask
        unique_slic = slic_mask.unique()
        for u in unique_slic[1:]:
            patch = slic_mask == u
            img_masked = image * patch
            img_patch = crop_to_nonzero(img_masked)
            label = self.model(img_patch)
            weedmap[patch] = label + 1
        return weedmap
        
    def forward(self, data_dict: DataDict):
        image = data_dict.image
        B, _, H, W = image.shape
        if self.training:
            return self.model(image)
        mask = self.plant_detector(data_dict.ndvi)
        slic = get_slic(image, self.slic_params)
        return self.segment(image, mask, slic)

    def get_learnable_params(self, train_params):
        return self.model.get_learnable_params(train_params)