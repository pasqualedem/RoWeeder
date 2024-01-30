import os
import torch
import torchvision

import numpy as np
import tifffile as tiff
from PIL import Image
from tqdm import tqdm
from einops import rearrange


def rotate_ortho(input_folder, output_folder, angle):
    os.makedirs(output_folder, exist_ok=True)
    channels_filename = os.listdir(input_folder)
    channels_path = [
        os.path.join(input_folder, channel) for channel in channels_filename
    ]
    ortho = []
    for channel in channels_path:
        if channel.endswith(".tif"):
            ortho.append(Image.fromarray(tiff.imread(channel)))
        else:
            ortho.append(Image.open(channel))

    rotated = [
        torchvision.transforms.functional.rotate(
            channel,
            angle,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            expand=True,
        )
        for channel in ortho
    ]

    # Crop black borders
    ground_truth = channels_filename.index("groundtruth.tif")
    cropped = []
    borders = None
    for i in range(len(rotated)):
        if i == ground_truth:
            continue
        channel, borders = crop_black_borders(rotated[i])
        cropped.append(channel)
    rotated_gt, _ = crop_black_borders(rotated[ground_truth], borders=borders)
    cropped.insert(ground_truth, rotated_gt)

    # Check shapes are equal
    shapes = [np.asarray(channel).shape[:2] for channel in cropped]
    assert len(set(shapes)) == 1, "Shapes are not equal"

    for name, channel in zip(channels_filename, cropped):
        name = os.path.splitext(name)[0]
        channel.save(os.path.join(output_folder, f"{name}.png"))
    print("Rotated images saved to: ", output_folder)


def crop_black_borders(image, borders=None):
    """
    Crop black borders from an image
    :param image: PIL image
    :return: PIL image
    """
    image = np.asarray(image)
    if borders is None:
        # If there is more than one channel, collapse along the channel axis to get 2D array
        grey_image = image
        if len(image.shape) == 3:
            grey_image = image.sum(axis=2)

        grey_image = np.asarray(grey_image)
        # Find non-black pixels
        non_black_pixels = np.where(image != 0)

        # Find the cropping boundaries
        top = np.min(non_black_pixels[0])
        bottom = np.max(non_black_pixels[0])
        left = np.min(non_black_pixels[1])
        right = np.max(non_black_pixels[1])
        borders = (top, bottom, left, right)
    else: # If borders are specified
        top, bottom, left, right = borders

    # Crop the image
    if len(image.shape) == 3:
        return Image.fromarray(image[top : bottom + 1, left : right + 1, :]), borders
    else:
        return Image.fromarray(image[top : bottom + 1, left : right + 1]), borders


def divide_ortho_into_patches(input_folder, output_folder, patch_size):
    """
    Divides the ortho images in the input folder into patches of the specified size
    and saves them in the output folder.

    Args:
        input_folder (str): Path to the folder containing the ortho images.
        output_folder (str): Path to the folder where the patches will be saved.
        patch_size (int): Size of the patches (both width and height).

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)
    channels_filename = os.listdir(input_folder)
    channels_path = [
        os.path.join(input_folder, channel) for channel in channels_filename
    ]
    ortho = [Image.open(channel) for channel in channels_path]
    tensors = [torchvision.transforms.PILToTensor()(channel) for channel in ortho]
    right_padding = patch_size - tensors[0].shape[1] % patch_size
    bottom_padding = patch_size - tensors[0].shape[2] % patch_size
    tensors = [
        torch.nn.functional.pad(
            tensor, (0, right_padding, 0, bottom_padding), mode="constant", value=0
        )
        for tensor in tensors
    ]

    patches = []
    for tensor in tensors:
        C, H, W = tensor.shape
        patch = torch.nn.functional.unfold(
            tensor.float(),
            kernel_size=patch_size,
            stride=patch_size,
        ).type(torch.uint8)
        patch = rearrange(patch, "(c p1 p2) n -> n c p1 p2", c=C, p1=patch_size, p2=patch_size)
        patches.append(patch)
        
    for name, channel in zip(channels_filename, patches):
        name = os.path.splitext(name)[0]
        print("Saving patches for channel: ", name)
        os.makedirs(os.path.join(output_folder, name), exist_ok=True)
        bar = tqdm(enumerate(channel), total=channel.shape[0])
        for i, patch in bar:
            patch = rearrange(patch, "c h w -> h w c").squeeze().numpy()
            patch = Image.fromarray(patch)
            patch.save(os.path.join(output_folder, name, f"{i}.png"))
    print("Patches saved to: ", output_folder)
