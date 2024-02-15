import torch
import torchvision.transforms as T

from selfweed.data.weedmap import WeedMapDataset


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
        
