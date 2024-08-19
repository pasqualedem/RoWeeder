from copy import deepcopy
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as T

from roweeder.data.weedmap import (
    SelfSupervisedWeedMapDataset,
    WeedMapDataset,
    ClassificationWeedMapDataset,
)


def get_preprocessing(dataset_params):
    preprocess_params = dataset_params.pop("preprocess")
    transforms = T.Compose(
        [
            lambda x: x.float() / 255.0,
            T.Normalize(
                mean=torch.tensor(preprocess_params["mean"]),
                std=torch.tensor(preprocess_params["std"]),
            ),
        ]
    )
    if "resize" in preprocess_params:
        transforms.transforms.insert(0, T.Resize(preprocess_params["resize"]))
    target_transforms = T.Compose(
        [
            T.Lambda(lambda x: x.long()),
        ]
    )
    deprocess = T.Compose(
        [
            T.Normalize(
                mean=[
                    -m / s
                    for m, s in zip(preprocess_params["mean"], preprocess_params["std"])
                ],
                std=[1 / s for s in preprocess_params["std"]],
            ),
            lambda x: x * 255.0,
        ]
    )
    return transforms, target_transforms, deprocess


def get_classification_dataloaders(dataset_params, dataloader_params, seed=42):
    dataset_params = deepcopy(dataset_params)
    transforms, target_transforms, deprocess = get_preprocessing(dataset_params)
    
    
    if "test_preprocess" in dataset_params:
        dataset_params["preprocess"] = dataset_params["test_preprocess"]
        dataset_params.pop("test_preprocess")
        test_transforms, test_target_transforms, test_deprocess = get_preprocessing(dataset_params)
    else:
        test_transforms = transforms
        test_target_transforms = target_transforms
        test_deprocess = deprocess

    if "train_fields" in dataset_params:
        train_params = deepcopy(dataset_params)
        train_params["fields"] = dataset_params["train_fields"]
        train_params.pop("train_fields")
        train_params.pop("test_fields")
        train_params.pop("test_root", None)
        train_set = ClassificationWeedMapDataset(
            **train_params,
            transform=transforms,
            target_transform=target_transforms,
        )
        val_set = ClassificationWeedMapDataset(
            **train_params,
            transform=transforms,
            target_transform=target_transforms,
        )
        index = train_set.index

        train_index, val_index = train_test_split(
            index, test_size=0.2, random_state=seed
        )
        train_set.index = train_index
        val_set.index = val_index

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=dataloader_params["batch_size"],
            shuffle=True,
            num_workers=dataloader_params["num_workers"],
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=dataloader_params["batch_size"],
            shuffle=False,
            num_workers=dataloader_params["num_workers"],
        )
    else:
        train_loader = None
        val_loader = None
    dataset_params["return_ndvi"] = True
    test_loader = get_testloader(
        dataset_params,
        dataloader_params,
        test_transforms,
        target_transforms=test_target_transforms,
    )
    return train_loader, val_loader, test_loader, deprocess


def get_dataloaders(dataset_params, dataloader_params, seed=42):
    if "gt_folder" not in dataset_params:
        # Classification dataset
        return get_classification_dataloaders(dataset_params, dataloader_params, seed)
    dataset_params = deepcopy(dataset_params)
    transforms, target_transforms, deprocess = get_preprocessing(dataset_params)

    if "train_fields" in dataset_params:
        train_params = deepcopy(dataset_params)
        train_params["fields"] = dataset_params["train_fields"]
        train_params.pop("train_fields")
        train_params.pop("test_fields")
        train_set = SelfSupervisedWeedMapDataset(
            **train_params,
            transform=transforms,
            target_transform=target_transforms,
        )
        val_set = WeedMapDataset(
            **train_params,
            transform=transforms,
            target_transform=target_transforms,
        )
        index = train_set.index

        train_index, val_index = train_test_split(
            index, test_size=0.2, random_state=seed
        )
        train_set.index = train_index
        val_set.index = val_index

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=dataloader_params["batch_size"],
            shuffle=True,
            num_workers=dataloader_params["num_workers"],
            collate_fn=train_set.collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=dataloader_params["batch_size"],
            shuffle=False,
            num_workers=dataloader_params["num_workers"],
        )
    else:
        train_loader = None
        val_loader = None
    test_loader = get_testloader(
        dataset_params, dataloader_params, transforms, target_transforms
    )

    return train_loader, val_loader, test_loader, deprocess


def get_testloader(dataset_params, dataloader_params, transforms, target_transforms):
    test_params = deepcopy(dataset_params)
    test_params["fields"] = dataset_params["test_fields"]
    test_params.pop("test_fields")
    test_params.pop("gt_folder", None)
    if "test_root" in dataset_params:
        test_params["root"] = dataset_params["test_root"]
        test_params.pop("test_root")
    if "train_fields" in dataset_params:
        test_params.pop("train_fields")

    test_set = WeedMapDataset(
        transform=transforms,
        target_transform=target_transforms,
        **test_params,
    )

    return torch.utils.data.DataLoader(
        test_set,
        batch_size=dataloader_params["batch_size"],
        shuffle=False,
        num_workers=dataloader_params["num_workers"],
    )


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
