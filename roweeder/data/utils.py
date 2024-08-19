from roweeder.utils.utils import EasyDict
from enum import Enum, StrEnum

import torch


class DataDict(EasyDict):
    image: torch.tensor
    target: torch.tensor
    name: str
    crops: torch.tensor
    weeds: torch.tensor
    ndvi: torch.tensor


class LABELS(Enum):
    BACKGROUND: int = 0
    WEED: int = 1
    CROP: int = 2
    
    
def pad_patches(patches: list):
    """
    Pad a list of patches to the same size.
    
    Args:
        patches (list): The patches to pad.
    """
    if patches[0].ndim == 3:
        ops = torch.cat
        patches = [patch.unsqueeze(0) for patch in patches]
    else:
        ops = torch.stack
    seq_flags = [torch.ones(1, patch.shape[0]) for patch in patches]
    max_seq_len = max(patch.shape[0] for patch in patches)
    max_height = max(patch.shape[2] for patch in patches)
    max_width = max(patch.shape[3] for patch in patches)
    padded_patches = []
    for i, patch in enumerate(patches):
        seq_len = patch.shape[0]
        height = patch.shape[2]
        width = patch.shape[3]
        pad_height = max_height - height
        pad_width = max_width - width
        back = 0
        front = max_seq_len - seq_len
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        padded_patch = torch.nn.functional.pad(patch, (left, right, top, bottom, 0, 0, back, front))
        seq_flags[i] = torch.nn.functional.pad(seq_flags[i], (back, front))
        padded_patches.append(padded_patch)
    return ops(padded_patches), ops(seq_flags)
    
    
def crop_to_nonzero(tensor: torch.Tensor):
    """
    Crop a tensor to the smallest bounding box around the non-zero elements.

    Args:
        tensor (torch.Tensor): The tensor to crop.
    """
    nonzeros = torch.nonzero(tensor)
    if tensor.ndim == 2:
        min_x = nonzeros[:, 1].min()
        max_x = nonzeros[:, 1].max()
        min_y = nonzeros[:, 0].min()
        max_y = nonzeros[:, 0].max()
        return tensor[min_y:max_y + 1, min_x:max_x + 1]
    if tensor.ndim == 3:
        min_x = nonzeros[:, 2].min()
        max_x = nonzeros[:, 2].max()
        min_y = nonzeros[:, 1].min()
        max_y = nonzeros[:, 1].max()
        return tensor[:, min_y:max_y + 1, min_x:max_x + 1]

    raise ValueError("Only 2D or 3D tensors are supported.")
    
    
def extract_plants(image: torch.Tensor, plant_mask: torch.Tensor):
    """
    Extract the individual plants from a patch.

    Args:
        patch (torch.Tensor): The patch to extract the plants from.
    """
    plant_ids = torch.unique(plant_mask)
    plant_ids = plant_ids[plant_ids != 0]
    if plant_ids.numel() == 0:
        return torch.empty(0, 3, 0, 0)
    plants = []

    for plant_id in plant_ids:
        plant = image * (plant_mask == plant_id)
        plants.append(crop_to_nonzero(plant))
    return pad_patches(plants)[0]