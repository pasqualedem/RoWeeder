import os
import torch
import torchvision

import numpy as np
from PIL import Image

def rotate_ortho(input_folder, output_folder, angle):
    channels_path = os.listdir(input_folder)
    channels_path = [os.path.join(input_folder, channel) for channel in channels_path]
    ortho = [np.array(Image.open(channel)) for channel in channels_path]
    ortho = torch.tensor(np.stack(ortho, axis=2))
    
    rotated = torchvision.transforms.functional.rotate(ortho, angle, mode='bilinear')
    
    for channel in range(rotated.shape[2]):
        img = Image.fromarray(rotated[:,:,channel].numpy())
        img.save(os.path.join(output_folder, f'{channel}.png'))