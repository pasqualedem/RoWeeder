import os
import math
import numpy as np
import cv2 as cv

from torchvision import transforms
from PIL import Image
from ezdl.datasets import WeedMapDataset

from detector import SplitLawinVegetationDetector, ModifiedHoughCropRowDetector

def get_drawn_img(img, theta_rho):
    draw_img = np.array(img[:3].transpose(1, 2, 0)).copy()
    for i in range(0, len(theta_rho)):
        rho = theta_rho[i][0]
        theta = theta_rho[i][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv.line(draw_img, pt1, pt2, (0,0,255), 1, cv.LINE_AA)
    return draw_img


def label(root, checkpoint, threshold=150):
    dest_root = "./masks"
    os.makedirs(dest_root, exist_ok=True)
    channels = ['R', 'G', 'B', 'NIR', 'RE']
    input_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = WeedMapDataset(root=root, channels=channels,
                                transform=input_transform, target_transform=lambda x: x,
                                return_path=True)
    labeler = SplitLawinVegetationDetector(checkpoint=checkpoint)
    detector = ModifiedHoughCropRowDetector(threshold=threshold)
    for img, target, path in dataset:
        mask = labeler(img)
        lines, components = detector(mask, return_components=True)
        blank = np.zeros_like(mask)
        line_mask = get_drawn_img(blank, lines)



        