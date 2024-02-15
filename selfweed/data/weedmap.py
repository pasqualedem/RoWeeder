import os
import torch
import torchvision

from torch.utils.data import Dataset


class WeedMapDataset(Dataset):
    def __init__(
        self,
        root,
        channels,
        transform=None,
        target_transform=None,
        return_path=False,
        gt_folder=None,
    ):
        super().__init__()
        self.root = root
        self.channels = channels
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path

        if gt_folder is not None:
            self.gt_folder = gt_folder
        else:
            self.gt_folder = os.path.join(self.root, "groundtruth")
        fields = os.listdir(self.gt_folder)
        self.channels_folder = [
            os.path.join(self.root, field, channel)
            for channel in self.channels
            for field in fields
        ]

    def __len__(self):
        return len(os.listdir(self.gt_folder))

    def __getitem__(self, i):
        gt_path = os.path.join(self.gt_folder, os.listdir(self.gt_folder)[i])
        gt = torchvision.io.read_image(gt_path)
        gt = gt[[2, 1, 0], ::]
        gt = gt.argmax(dim=0)
        gt = self.target_transform(gt)

        channels = []
        for channel_folder in self.channels_folder:
            channel_path = os.path.join(channel_folder, os.listdir(channel_folder)[i])
            channel = torchvision.io.read_image(channel_path)
            channels.append(channel)
        channels = torch.cat(channels).float()
        channels = self.transform(channels)

        return (
            (channels, gt, {"input_name": gt_path})
            if self.return_path
            else (channels, gt)
        )
