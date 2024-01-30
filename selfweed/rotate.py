import os
import torch
import torchvision

import numpy as np
from PIL import Image


def rotate_ortho(input_folder, output_folder, angle):
    os.makedirs(output_folder, exist_ok=True)
    channels_filename = os.listdir(input_folder)
    channels_path = [os.path.join(input_folder, channel) for channel in channels_filename]
    ortho = [Image.open(channel) for channel in channels_path]

    rotated = [torchvision.transforms.functional.rotate(
        channel, angle, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, expand=True
    ) for channel in ortho]
    
    # Crop black borders
    rotated = [crop_black_borders(channel) for channel in rotated]
    
    # Check shapes are equal
    shapes = [np.asarray(channel).shape[:2] for channel in rotated]
    assert len(set(shapes)) == 1, "Shapes are not equal"

    for name, channel in zip(channels_filename, rotated):
        channel.save(os.path.join(output_folder, name))
    print("Rotated images saved to: ", output_folder)
    
    
def crop_black_borders(image):
    """
    Crop black borders from an image
    :param image: PIL image
    :return: PIL image
    """
    image = np.asarray(image)
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

    # Crop the image
    if len(image.shape) == 3:
        return Image.fromarray(image[top:bottom + 1, left:right + 1, :])
    else:
        return Image.fromarray(image[top:bottom + 1, left:right + 1])

