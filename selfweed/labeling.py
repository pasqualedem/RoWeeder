import os
import math
import torch
import numpy as np
import cv2

from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import yaml
from selfweed.data import get_dataset

from selfweed.detector import (
    HoughCropRowDetector,
    HoughDetectorDict,
    get_vegetation_detector,
)


def get_drawn_img(img, theta_rho, color=(255, 255, 255)):
    """
    Draws lines on an image based on the given theta-rho parameters.

    Args:
        img (numpy.ndarray): The input image.
        theta_rho (list): List of theta-rho parameters for drawing lines.
        color (tuple, optional): The color of the lines. Defaults to (255, 255, 255).

    Returns:
        numpy.ndarray: The image with lines drawn on it.
    """
    draw_img = np.array(img[:3].transpose(1, 2, 0)).copy()
    draw_img = draw_img.astype(np.uint8)
    for i in range(len(theta_rho)):
        rho = theta_rho[i][0]
        theta = theta_rho[i][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(draw_img, pt1, pt2, color, 2, cv2.LINE_AA)
    return draw_img


def label_from_row(mask, row_image):
    conn_components = cv2.connectedComponents(mask.cpu().numpy().astype(np.uint8))[1]
    conn_components = torch.tensor(conn_components)
    row_crop_intersection = conn_components * row_image.bool()
    crop_values = row_crop_intersection.unique()
    if len(crop_values) == 1:
        return torch.stack([~mask, torch.zeros_like(mask), mask])
    # Remove zeros
    crop_values = crop_values[1:]
    crop_mask = torch.isin(conn_components, crop_values)
    crops = conn_components * crop_mask
    weeds = conn_components * (~crop_mask)
    background = conn_components == 0
    return torch.stack([background, crops, weeds])


def out_summary_file(
    outfile,
    parameters
):
    with open(outfile, "w") as f:
        # Write parameters to file
        yaml.dump(parameters, f)


def gt_defix(gt):
    gt[gt == 1] = 10000
    return gt


def load_and_label(outdir, param_file, interactive=True):
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)
    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    param_id = param_file.split("/")[-1].split(".")[0]
    outsubdir = os.path.join(outdir, f"{now}_{param_id}")
    for _ in label(outsubdir, **params, interactive=interactive):
        pass


def save_and_label(
    outdir,
    dataset_params,
    plant_detector_params,
    hough_detector_params,
    interactive=False,
):
    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    hashid = hash(now)
    hashid_8 = str(hashid)[-8:]
    outsubdir = os.path.join(outdir, f"{now}_{hashid_8}")
    os.makedirs(outdir, exist_ok=True)

    out_summary_file(
        f"{outdir}/{hashid_8}.yaml",
        {
            "dataset_params": dataset_params,
            "plant_detector_params": plant_detector_params,
            "hough_detector_params": hough_detector_params,
        }
    )
    yield from label(
        outsubdir,
        dataset_params,
        plant_detector_params,
        hough_detector_params,
        interactive,
    )


def label(
    outdir,
    dataset_params,
    plant_detector_params,
    hough_detector_params,
    interactive=False,
):
    os.makedirs(outdir, exist_ok=True)
    
    plant_detector = get_vegetation_detector(plant_detector_params['name'], plant_detector_params['params'])
    detector = HoughCropRowDetector(
        **hough_detector_params,
        crop_detector=plant_detector,
    )
    dataset = get_dataset(**dataset_params)
    print(len(dataset))
    for i, (img, target, additional) in enumerate(tqdm(dataset)):
        mask = plant_detector(img)
        result_dict = detector.predict_from_mask(mask)
        lines = result_dict[HoughDetectorDict.LINES]
        blank = mask.cpu().numpy().astype(np.uint8)
        line_mask = get_drawn_img(
            torch.zeros_like(torch.tensor(blank)).numpy(), lines, color=(255, 0, 255)
        )
        argmask = mask[0].type(torch.uint8)
        weed_map = label_from_row(argmask, torch.tensor(line_mask).permute(2, 0, 1)[0])
        weed_map = weed_map.argmax(dim=0)
        weed_map = weed_map.cpu().numpy().astype(np.uint8)
        path, basename = os.path.split(additional["input_name"])
        path, gt_folder = os.path.split(path)
        path, field = os.path.split(path)
        img_out_path = os.path.join(
            outdir, field, basename
        )
        cv2.imwrite(img_out_path, weed_map)
        if interactive:
            yield i
