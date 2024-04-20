from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as T

from selfweed.data.weedmap import SelfSupervisedWeedMapDataset, WeedMapDataset


def get_dataloaders(dataset_params, dataloader_params, seed=42):
    transforms = T.Compose(
        [
            T.Normalize(
                mean=torch.tensor(dataset_params["mean"]),
                std=torch.tensor(dataset_params["std"]),
            ),
        ]
    )
    target_transforms = T.Compose(
        [
            T.Lambda(lambda x: x.long()),
        ]
    )

    train_set = SelfSupervisedWeedMapDataset(
        channels=dataset_params["channels"],
        root=dataset_params["root"],
        gt_folder=dataset_params["gt_folder"],
        fields=dataset_params["train_fields"],
        transform=transforms,
        target_transform=target_transforms,
    )
    val_set = WeedMapDataset(
        channels=dataset_params["channels"],
        root=dataset_params["root"],
        gt_folder=dataset_params["gt_folder"],
        fields=dataset_params["train_fields"],
        transform=transforms,
        target_transform=target_transforms,
    )
    index = train_set.index

    train_index, val_index = train_test_split(index, test_size=0.2, random_state=seed)
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
    test_loader = get_testloader(dataset_params, dataloader_params)
    
    deprocess = T.Compose(
        [
            T.Normalize(
                mean=[-m / s for m, s in zip(dataset_params["mean"], dataset_params["std"])],
                std=[1 / s for s in dataset_params["std"]],
            ),
        ]
    )

    return train_loader, val_loader, test_loader, deprocess


def get_testloader(dataset_params, dataloader_params):
    transforms = T.Compose(
        [
            T.Normalize(mean=dataset_params["mean"], std=dataset_params["std"]),
        ]
    )
    target_transforms = T.Compose(
        [
            T.Lambda(lambda x: x.long()),
        ]
    )

    test_set = WeedMapDataset(
        root=dataset_params["root"],
        channels=dataset_params["channels"],
        transform=transforms,
        target_transform=target_transforms,
        fields=dataset_params["test_fields"],
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
