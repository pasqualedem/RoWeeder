import itertools
import os
import torch
import torchvision

from torch.utils.data import Dataset


class WeedMapDataset(Dataset):
    def __init__(
        self,
        root,
        channels,
        fields,
        gt_folders=None,
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
        self.gt_folders = {
            field: os.path.join(self.root, field, "groundtruth")
            for field in self.fields
        } if gt_folders is None else gt_folders
        lengths = [len(os.listdir(gt_folder)) for gt_folder in self.gt_folders.values()]
        cum_lengths = [0] + list(itertools.accumulate(lengths))
        print(cum_lengths)
        self.index_to_field = {
            index: (field, start_index)
            for field, start_index, end_index in zip(
                fields, cum_lengths[:-1], cum_lengths[1:]
            )
            for index in range(start_index, end_index)
        }
        self.cum = 0

    def __len__(self):
        return len(self.index_to_field)

    def __getitem__(self, i):
        print(i)
        self.cum += 1
        field, start_index = self.index_to_field[i]
        gt_path = os.path.join(
            self.gt_folders[field], os.listdir(self.gt_folders[field])[i - start_index]
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
                os.listdir(os.path.join(self.root, field, channel_folder))[i - start_index],
            )
            channel = torchvision.io.read_image(channel_path)
            channels.append(channel)
        channels = torch.cat(channels).float()
        channels = self.transform(channels)

        return (
            (channels, gt, {"input_name": gt_path})
            if self.return_path
            else (channels, gt)
        )
