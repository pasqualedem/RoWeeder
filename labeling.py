import os
import math
import numpy as np
import cv2

from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from ezdl.datasets import WeedMapDataset

from detector import HoughCropRowDetector, SplitLawinVegetationDetector, ModifiedHoughCropRowDetector

def get_drawn_img(img, theta_rho, color=(255, 255, 255)):
    print(img.shape)
    draw_img = np.array(img[:3].transpose(1, 2, 0)).copy()
    draw_img = draw_img.astype(np.uint8)
    for i in range(0, len(theta_rho)):
        rho = theta_rho[i][0]
        theta = theta_rho[i][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(draw_img, pt1, pt2, color, 1, cv2.LINE_AA)
    return draw_img


def label(root, outdir, checkpoint, threshold=150):
    os.makedirs(outdir, exist_ok=True)
    channels = ['R', 'G', 'B', 'NIR', 'RE']
    input_transform = lambda x: x
    dataset = WeedMapDataset(root=root, channels=channels,
                                transform=input_transform, target_transform=lambda x: x,
                                return_path=True)
    labeler = SplitLawinVegetationDetector(checkpoint_path=checkpoint)
    detector = HoughCropRowDetector(threshold=threshold, crop_detector=labeler)
    for img, target, path in tqdm(dataset):
        mask = labeler(img)
        lines, components = detector.predict_from_mask(mask, return_components=True)
        blank = np.zeros_like(mask.cpu())
        line_mask = get_drawn_img(blank, lines)
        cv2.imwrite(os.path.join(outdir, os.path.basename(path['input_name'])), line_mask)



        