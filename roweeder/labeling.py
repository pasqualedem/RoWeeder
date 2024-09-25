from copy import deepcopy
import os
import math
import torch
import numpy as np
import skimage as ski
import cv2

from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import yaml
from roweeder.data import get_dataset
from roweeder.data.utils import DataDict, crop_to_nonzero

from roweeder.detector import (
    HoughCropRowDetector,
    HoughDetectorDict,
    get_vegetation_detector,
)
from roweeder.visualize import map_grayscale_to_rgb


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
        cv2.line(draw_img, pt1, pt2, color, 4, cv2.LINE_AA)
    return draw_img


def get_slic(img, slic_params):
    """
    Get the SLIC segmentation of an image.

    Args:
        img (numpy.ndarray): The input image.
        slic_params (dict): Parameters for the SLIC segmentation.

    Returns:
        numpy.ndarray: The SLIC segmentation.
    """
    img = img.permute(1, 2, 0).cpu().numpy()
    slic_params_cp = deepcopy(slic_params)
    N = int(np.prod(img.shape[:-1]) * slic_params_cp.pop("percent"))
    slic_params_cp["n_segments"] = N
    slic = ski.segmentation.slic(img[:, :, :3], **slic_params_cp)
    return slic


def get_patches(img, weedmap, slic_map):
    """
    Get the patches of the image based on the SLIC segmentation and their class

    Args:
        img (numpy.ndarray): The input image.
        weedmap (numpy.ndarray): The weed map. 0 for background, 1 for crop, 2 for weed.
        slic_map (numpy.ndarray): The SLIC segmentation.

    Returns:
        list: List of patches.
    """
    MIN_HEIGHT = 10
    MIN_WIDTH = 10
    MIN_PLANT_PERCENT = 0.1
    patches = []
    slic_map = torch.tensor(slic_map)
    weedmap = weedmap.argmax(dim=0).cpu()
    weedmap_slic = torch.zeros_like(slic_map)
    for i in np.unique(slic_map):
        mask = slic_map == i
        plant_mask = mask * weedmap
        if plant_mask.sum() == 0:
            continue
        patch_mask = crop_to_nonzero(plant_mask)
        min_size = (
            patch_mask.shape[0] >= MIN_HEIGHT
            and patch_mask.shape[1] >= MIN_WIDTH
        )
        values, counts = torch.unique(patch_mask, return_counts=True)
        complete_counts = torch.zeros(3, dtype=int)
        complete_counts[values] = counts
        if values.sum() == 0:
            continue
        label = complete_counts[1:].argmax() + 1
        min_percent = complete_counts[label] >= (MIN_PLANT_PERCENT * complete_counts.sum())
        patch = crop_to_nonzero(mask * img)
        weedmap_slic[plant_mask.bool()] = label
        if min_size and min_percent:
            patches.append((patch, label-1))
    return weedmap_slic, patches


def label_from_row(img, mask, row_image, slic_params=None):
    conn_components = cv2.connectedComponents(mask.cpu().numpy().astype(np.uint8))[1]
    conn_components = torch.tensor(conn_components)
    row_crop_intersection = conn_components * row_image.bool()
    crop_values = row_crop_intersection.unique()
    if len(crop_values) == 1:
        weedmap = torch.stack([~mask, torch.zeros_like(mask), mask])
        if slic_params is None:
            return weedmap, None, []
        return weedmap, *slic_label(img, slic_params, weedmap)
    # Remove zeros
    crop_values = crop_values[1:]
    crop_mask = torch.isin(conn_components, crop_values)
    crops = conn_components * crop_mask
    weeds = conn_components * (~crop_mask)
    background = conn_components == 0
    weedmap = torch.stack([background, crops, weeds])
    if slic_params is None:
        return weedmap, None, []
    return weedmap, *slic_label(img, slic_params, weedmap)


# TODO Rename this here and in `label_from_row`
def slic_label(img, slic_params, weedmap):
    slic = get_slic(img, slic_params)
    weedmap_slic, patches = get_patches(img, weedmap, slic)
    return weedmap_slic, patches


def out_summary_file(outfile, parameters):
    with open(outfile, "w") as f:
        # Write parameters to file
        yaml.dump(parameters, f)


def gt_defix(gt):
    gt[gt == 1] = 10000
    return gt


def get_line_mask(lines, shape):
    color = 1
    blank = np.zeros(shape, dtype=np.uint8)
    line_mask = get_drawn_img(
        blank, lines, color=color
    )
    return np.moveaxis(line_mask, 2, 0)[0]
    
    
def get_on_off_row_plants(mask, row_image):
    if len(mask.shape) == 3:
        mask = mask[0]
    conn_components = cv2.connectedComponents(mask.cpu().numpy().astype(np.uint8))[1]
    conn_components = torch.tensor(conn_components).to(mask.device)
    all_plants = conn_components.unique()
    if len(all_plants) == 1:
        return torch.zeros_like(mask), torch.zeros_like(mask)
    row_plant_intersection = conn_components * row_image.bool()
    on_row_plants = row_plant_intersection.unique()
    # Remove zeros
    on_row_plants = on_row_plants[1:]
    plants_mask = torch.isin(conn_components, on_row_plants)
    on_row_plants_mask = (conn_components * plants_mask).bool()
    off_row_plants_mask = (conn_components * (~plants_mask)).bool()
    return on_row_plants_mask, off_row_plants_mask
    

def load_and_label(outdir, param_file, interactive=True):
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)
    param_id = param_file.split("/")[-1].split(".")[0]
    outsubdir = os.path.join(outdir, param_id)
    for _ in label(outsubdir, **params, interactive=interactive):
        pass


def save_and_label(
    outdir,
    dataset_params,
    plant_detector_params,
    hough_detector_params,
    slic_params,
    interactive=False,
):
    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    hashid = hash(now)
    hashid_8 = str(hashid)[-8:]
    outsubdir = os.path.join(outdir, hashid_8)
    os.makedirs(outdir, exist_ok=True)

    out_summary_file(
        f"{outdir}/{hashid_8}.yaml",
        {
            "dataset_params": dataset_params,
            "plant_detector_params": plant_detector_params,
            "hough_detector_params": hough_detector_params,
            "slic_params": slic_params,
        },
    )
    yield from label(
        outsubdir,
        dataset_params,
        plant_detector_params,
        hough_detector_params,
        slic_params,
        interactive,
    )


def label(
    outdir,
    dataset_params,
    plant_detector_params,
    hough_detector_params,
    slic_params,
    interactive=False,
):
    os.makedirs(outdir, exist_ok=True)
    channels = ["R", "G", "B", "NIR", "RE"]
    gt_outdir = os.path.join(outdir, "pseudogt")
    gt_slic_outdir = os.path.join(outdir, "pseudogt_slic")
    patches_outdir = os.path.join(outdir, "patches")
    os.makedirs(patches_outdir, exist_ok=True)
    os.makedirs(gt_outdir, exist_ok=True)

    plant_detector = get_vegetation_detector(
        plant_detector_params["name"], plant_detector_params["params"]
    )
    detector = HoughCropRowDetector(
        **hough_detector_params,
        crop_detector=plant_detector,
    )
    dataset = get_dataset(**dataset_params)
    for i, (data_dict) in enumerate(tqdm(dataset)):
        img = data_dict.image
        gt = data_dict.target
        mask = plant_detector(img)
        result_dict = detector.predict_from_mask(mask)
        lines = result_dict[HoughDetectorDict.LINES]
        blank = mask.cpu().numpy().astype(np.uint8)
        line_mask = get_drawn_img(
            torch.zeros_like(torch.tensor(blank)).numpy(), lines, color=(255, 0, 255)
        )
        argmask = mask[0].type(torch.uint8)
        weed_map, weed_map_slic, patches = label_from_row(
            img,
            argmask,
            torch.tensor(line_mask).permute(2, 0, 1)[0],
            slic_params=slic_params,
        )
        weed_map = weed_map.argmax(dim=0)
        weed_map = weed_map.cpu().numpy().astype(np.uint8)
        weed_map = map_grayscale_to_rgb(
            weed_map, mapping={1: (0, 255, 0), 2: (255, 0, 0)}
        ).transpose(2, 0, 1)
        weed_map_slic = weed_map_slic.cpu().numpy().astype(np.uint8)
        weed_map_slic = map_grayscale_to_rgb(
            weed_map_slic, mapping={1: (0, 255, 0), 2: (255, 0, 0)}
        ).transpose(2, 0, 1)
        # RGB to BGR
        weed_map = np.moveaxis(weed_map[[2, 1, 0], ::], 0, 2)
        weed_map_slic = np.moveaxis(weed_map_slic[[2, 1, 0], ::], 0, 2)
        path, basename = os.path.split(data_dict.name)
        filename, _ = os.path.splitext(basename)
        path, gt_folder = os.path.split(path)
        path, field = os.path.split(path)
        os.makedirs(os.path.join(gt_outdir, field), exist_ok=True)
        os.makedirs(os.path.join(gt_slic_outdir, field), exist_ok=True)
        for ch in channels + ["RGB"]:
            os.makedirs(os.path.join(patches_outdir, field, ch), exist_ok=True)
        for i, (patch, label) in enumerate(patches):
            patch = (patch * 255).type(torch.uint8)
            for j, ch in enumerate(channels):
                patch_out_path = os.path.join(
                    patches_outdir, field, ch, f"{filename}_{i}_{label}.png"
                )
                cv2.imwrite(patch_out_path, patch[j].numpy())
            patch_rgb_out_path = os.path.join(
                patches_outdir, field, "RGB", f"{filename}_{i}_{label}.png"
            )
            cv2.imwrite(patch_rgb_out_path, np.moveaxis(patch[:3].numpy(), 0, 2))
        img_out_path = os.path.join(gt_outdir, field, basename)
        img_out_path_slic = os.path.join(gt_slic_outdir, field, basename)
        cv2.imwrite(img_out_path, weed_map)
        cv2.imwrite(img_out_path_slic, weed_map_slic)
        if interactive:
            yield i
