import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FV

from roweeder.data.utils import DataDict, crop_to_nonzero
from roweeder.detector import HoughDetectorDict
from roweeder.labeling import get_drawn_img, get_slic, label_from_row
from roweeder.models.utils import ModelOutput

class HoughSLICSegmentationWrapper(nn.Module):
    classificator_size = (224, 224)
    def __init__(self, classification_model, plant_detector, slic_params, use_ndvi=True, internal_batch_size=1) -> None:
        super().__init__()
        self.model = classification_model
        self.plant_detector = plant_detector
        self.slic_params = slic_params
        self.use_ndvi = use_ndvi
        self.internal_batch_size = internal_batch_size
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
    
    
class HoughCC(nn.Module):
    def __init__(self, hough_detector, plant_detector, use_ndvi=True) -> None:
        super().__init__()
        self.hough_detector = hough_detector
        self.plant_detector = plant_detector
        self.use_ndvi = use_ndvi
        self.__repr__ = f"HoughCC:\n{self.hough_detector.__repr__}"
        
    def segment(self, image, mask):
        result_dict = self.hough_detector.predict_from_mask(mask)
        lines = result_dict[HoughDetectorDict.LINES]
        conn_components = cv2.connectedComponents(mask[0].cpu().numpy().astype(np.uint8))[1]
        conn_components = torch.tensor(conn_components)
        blank = mask.cpu().numpy().astype(np.uint8)
        line_mask = get_drawn_img(
            torch.zeros_like(torch.tensor(blank)).numpy(), lines, color=(255, 0, 255)
        )
        row_image = torch.tensor(line_mask).permute(2, 0, 1)[0]
        row_crop_intersection = conn_components * row_image.bool()
        crop_values = row_crop_intersection.unique()
        if len(crop_values) == 1:
            return torch.cat([~mask, torch.zeros_like(mask), mask])
        # Remove zeros
        crop_values = crop_values[1:]
        crop_mask = torch.isin(conn_components, crop_values)
        crops = conn_components * crop_mask
        weeds = conn_components * (~crop_mask)
        background = conn_components == 0
        return torch.stack([background, crops, weeds]).float().to(mask.device)

    def forward(self, image, ndvi=None):
        B, _, H, W = image.shape
        segmentations = []
        for i in range(image.shape[0]):
            mask = self.plant_detector(ndvi=ndvi[i])[0]
            segmentations.append(self.segment(image[i], mask))
        return ModelOutput(logits=torch.cat(segmentations), scores=None)

    def get_learnable_params(self, train_params):
        return self.model.get_learnable_params(train_params)
    
    
class HoughSLIC(nn.Module):
    def __init__(self, hough_detector, plant_detector, slic_params, use_ndvi=True) -> None:
        super().__init__()
        self.hough_detector = hough_detector
        self.plant_detector = plant_detector
        self.slic_params = slic_params
        self.use_ndvi = use_ndvi
        self.__repr__ = f"HoughCC:\n{self.hough_detector.__repr__}"
        
    def segment(self, image, mask, slic):
        mask = mask.bool()
        weedmap = mask.clone().long()
        slic_mask = slic * mask if self.use_ndvi else slic
        unique_slic = slic_mask.unique()
        for u in unique_slic[1:]:
            patch = slic_mask == u
            plant_mask = mask * weedmap
            patch_mask = crop_to_nonzero(plant_mask)
            values, counts = torch.unique(patch_mask, return_counts=True)
            complete_counts = torch.zeros(3, dtype=int, device=mask.device)
            complete_counts[values.long()] = counts   
            if values.sum() == 0:
                continue
            label = complete_counts[1:].argmax() + 1
            weedmap[patch] = label
        return F.one_hot(weedmap, num_classes=3).permute(0, 3, 1, 2).float()

    def forward(self, image, ndvi=None):
        B, _, H, W = image.shape
        segmentations = []
        for i in range(image.shape[0]):
            mask = self.plant_detector(ndvi=ndvi[i])[0]
            slic = torch.tensor(get_slic(image[i], self.slic_params), device=mask.device)
            segmentations.append(self.segment(image[i], mask, slic))
        return ModelOutput(logits=torch.cat(segmentations), scores=None)

    def get_learnable_params(self, train_params):
        return self.model.get_learnable_params(train_params)