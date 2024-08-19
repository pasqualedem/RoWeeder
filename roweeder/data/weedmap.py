import itertools
import os
import torch
import torchvision
import cv2

from torch.utils.data import Dataset

from roweeder.data.utils import DataDict, extract_plants, LABELS, pad_patches


class WeedMapDataset(Dataset):
    id2class = {
        0: "background",
        1: "crop",
        2: "weed",
    }
    def __init__(
        self,
        root,
        channels,
        fields,
        gt_folder=None,
        transform=None,
        target_transform=None,
        return_path=False,
        return_ndvi=False, # Return NDVI as extra channel
    ):
        super().__init__()
        self.root = root
        self.channels = channels
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path
        self.fields = fields
        self.return_ndvi = return_ndvi

        self.channels = channels
        if gt_folder is None:
            self.gt_folders = {
                field: os.path.join(self.root, field, "groundtruth")
                for field in self.fields
            }
        else:
            self.gt_folders = {
                field: os.path.join(gt_folder, field) for field in self.fields
            }
            for k, v in self.gt_folders.items():
                if os.path.isdir(os.path.join(v, os.listdir(v)[0])):
                    self.gt_folders[k] = os.path.join(v, "groundtruth") 
            
        self.index = [
            (field, filename) for field in self.fields for filename in os.listdir(self.gt_folders[field])
        ]

    def __len__(self):
        return len(self.index)
    
    def _get_gt(self, gt_path):
        gt = torchvision.io.read_image(gt_path)
        gt = gt[[2, 1, 0], ::]
        gt = gt.argmax(dim=0)
        gt = self.target_transform(gt)
        return gt
    
    def _get_image(self, field, filename):
        channels = []
        for channel_folder in self.channels:
            channel_path = os.path.join(
                self.root,
                field,
                channel_folder,
                filename
            )
            channel = torchvision.io.read_image(channel_path)
            channels.append(channel)
        channels = torch.cat(channels).float()
        return self.transform(channels)

    def _get_ndvi(self, field, filename):
        nir_red_path = [
            os.path.join(
                self.root,
                field,
                ch,
                filename
            ) for ch in ["NIR", "R"]
        ]
        nir_red = [torchvision.io.read_image(channel_path).float() for channel_path in nir_red_path]
        ndvi = (nir_red[0] - nir_red[1]) / (nir_red[0] + nir_red[1])
        # Replaces NaN values with 0
        ndvi[torch.isnan(ndvi)] = 0
        return ndvi

    def __getitem__(self, i):
        field, filename = self.index[i]
        gt_path = os.path.join(
            self.gt_folders[field], filename
        )
        gt = self._get_gt(gt_path)
        channels = self._get_image(field, filename)

        data_dict = DataDict(
            image = channels,
            target = gt,
        )
        if self.return_path:
            data_dict.name = gt_path
        
        if self.return_ndvi:
            ndvi = self._get_ndvi(field, filename)
            data_dict.ndvi = ndvi
        return data_dict
    
    
class ClassificationWeedMapDataset(Dataset):
    id2class = {
        0: "crop",
        1: "weed",
    }
    def __init__(
        self,
        root,
        channels,
        fields,
        transform=None,
        target_transform=None,
        return_path=False,
    ):
        super().__init__()
        self.root = root
        self.channels = channels
        self.transform = transform
        self.return_path = return_path
        self.target_transform = target_transform
        self.fields = fields

        self.channels = channels
            
        self.index = [
            (field, filename) for field in self.fields for filename in os.listdir(os.path.join(self.root, field, channels[0]))
        ]
        
    def _get_image(self, field, filename):
        channels = []
        for channel_folder in self.channels:
            channel_path = os.path.join(
                self.root,
                field,
                channel_folder,
                filename
            )
            channel = torchvision.io.read_image(channel_path)
            channels.append(channel)
        channels = torch.cat(channels).float()
        return self.transform(channels)
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, i):
        field, filename = self.index[i]
        channels = self._get_image(field, filename)
        fname = os.path.splitext(filename)[0]
        label = int(fname.split("_")[-1])
        data_dict = DataDict(
            image = channels,
            target = self.target_transform(torch.tensor(label))
        )
        if self.return_path:
            data_dict.name = os.path.join(self.root, field, filename)
        return data_dict


class SelfSupervisedWeedMapDataset(WeedMapDataset):
    def __init__(self, root, channels, fields, gt_folder=None, transform=None, target_transform=None, return_path=False, max_plants=10):
        super().__init__(root, channels, fields, gt_folder, transform, target_transform, return_path)
        self.max_plants = max_plants
    def __getitem__(self, i):
        data_dict = super().__getitem__(i)
        
        connected_components = torch.tensor(cv2.connectedComponents(data_dict.target.numpy().astype('uint8'))[1])
        crops_mask = data_dict.target == LABELS.CROP.value
        crops_mask = connected_components * crops_mask
        crops_mask = extract_plants(data_dict.image, crops_mask)
        if crops_mask.shape[0] > self.max_plants:
            # Get self.max_plants random crops
            indices = torch.randperm(crops_mask.shape[0])[:self.max_plants]
            crops_mask = crops_mask[indices]
        
        weeds_mask = data_dict.target == LABELS.WEED.value
        weeds_mask = connected_components * weeds_mask
        weeds_mask = extract_plants(data_dict.image, weeds_mask)
        if weeds_mask.shape[0] > self.max_plants:
            # Get self.max_plants random weeds
            indices = torch.randperm(weeds_mask.shape[0])[:self.max_plants]
            weeds_mask = weeds_mask[indices]
        data_dict.crops = crops_mask
        data_dict.weeds = weeds_mask
        return data_dict
        
    def collate_fn(self, batch):
        crops = [item.crops for item in batch]
        weeds = [item.weeds for item in batch]
        crops = pad_patches(crops)
        weeds = pad_patches(weeds)
        return DataDict(
            image=torch.stack([item.image for item in batch]),
            target=torch.stack([item.target for item in batch]),
            crops=crops,
            weeds=weeds,
        )
        