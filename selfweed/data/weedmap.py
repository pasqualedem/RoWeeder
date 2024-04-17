import itertools
import os
import torch
import torchvision
import cv2

from torch.utils.data import Dataset

from selfweed.data.utils import DataDict, extract_plants, LABELS, pad_patches


class WeedMapDataset(Dataset):
    id2label = {
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
    ):
        super().__init__()
        self.root = root
        self.channels = channels
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path
        self.fields = fields

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
            
        self.index = [
            (field, filename) for field in self.fields for filename in os.listdir(self.gt_folders[field])
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        field, filename = self.index[i]
        gt_path = os.path.join(
            self.gt_folders[field], filename
        )
        gt = torchvision.io.read_image(gt_path)
        gt = gt[[2, 1, 0], ::]
        gt = gt.argmax(dim=0)
        gt = self.target_transform(gt)

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
        channels = self.transform(channels)

        data_dict = {
            DataDict.IMAGE: channels,
            DataDict.TARGET: gt,
        }
        if self.return_path:
            data_dict[DataDict.NAME] = gt_path
        return data_dict


class SelfSupervisedWeedMapDataset(WeedMapDataset):
    def __getitem__(self, i):
        data_dict = super().__getitem__(i)
        
        connected_components = torch.tensor(cv2.connectedComponents(data_dict[DataDict.TARGET].numpy().astype('uint8'))[1])
        crops_mask = data_dict[DataDict.TARGET] == LABELS.CROP.value
        crops_mask = connected_components * crops_mask
        crops_mask = extract_plants(data_dict[DataDict.IMAGE], crops_mask)
        
        weeds_mask = data_dict[DataDict.TARGET] == LABELS.WEED.value
        weeds_mask = connected_components * weeds_mask
        weeds_mask = extract_plants(data_dict[DataDict.IMAGE], weeds_mask)
        
        return {
            **data_dict,
            DataDict.CROPS: crops_mask,
            DataDict.WEEDS: weeds_mask,
        }
        
    def collate_fn(self, batch):
        crops = [item[DataDict.CROPS] for item in batch]
        weeds = [item[DataDict.WEEDS] for item in batch]
        crops = pad_patches(crops)
        weeds = pad_patches(weeds)
        return {
            DataDict.IMAGE: torch.stack([item[DataDict.IMAGE] for item in batch]),
            DataDict.TARGET: torch.stack([item[DataDict.TARGET] for item in batch]),
            DataDict.CROPS: crops,
            DataDict.WEEDS: weeds,
        }
        