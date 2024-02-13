import torch

from selfweed.data.weedmap import WeedMapDataset


def get_dataloaders(params):

    train_set = WeedMapDataset(
        root=params["root"],
        split=params["split"],
        transform=params["transform"],
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
        
