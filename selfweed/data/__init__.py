import torch
import torchvision.transforms as T

from selfweed.data.weedmap import WeedMapDataset
# from ezdl.datasets import WeedMapDataset as WeedOldDataset


def get_trainval(dataset_params, dataloader_params):
    transforms  = T.Compose([
        T.Normalize(mean=dataset_params["mean"], std=dataset_params["std"]),
    ])
    target_transforms = T.Compose([
        T.Lambda(lambda x: x.long()),
    ])
    

    train_set = WeedMapDataset( 
        channels=dataset_params["channels"],
        root=dataset_params["root"],
        gt_folder=dataset_params["gt_folder"],
        fields=dataset_params["fields"],
        transform=transforms,
        target_transform=target_transforms,
    )
    
    train_set, val_set = torch.utils.data.random_split(
        train_set,
        [round(len(train_set) * 0.8), round(len(train_set) * 0.2)],
    )
    
    return train_set, val_set
        

def get_dataset(root, modality, fields):
    dataclass = WeedMapDataset
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