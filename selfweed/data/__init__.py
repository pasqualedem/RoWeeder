import torch
import torchvision.transforms as T

from selfweed.data.weedmap import WeedMapDataset
from ezdl.datasets import WeedMapDataset as WeedOldDataset


def get_dataloaders(params):
    transforms  = T.Compose([
        T.Normalize(mean=params["mean"], std=params["std"]),
    ])
    

    train_set = WeedMapDataset( 
        channels=params["channels"],
        root=params["root"],
        gt_folder=params["gt_folder"],
        transform=transforms,
    )
    
    train_set, val_set = torch.utils.data.random_split(
        train_set,
        [int(len(train_set) * 0.8), int(len(train_set) * 0.2)],
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
    )
    return train_loader, val_loader
        

def get_dataset(root, modality, fields):
    dataclass = WeedOldDataset if modality == "Old Dataset" else WeedMapDataset
    channels = ["R", "G", "B", "NIR", "RE"]
    input_transform = lambda x: x / 255.0
    
    args = dict(
        root=root,
        channels=channels,
        transform=input_transform,
        target_transform=lambda x: x,
        return_path=True,
    )
    if modality == "New Dataset":
        args["fields"] = fields

    return dataclass(**args)