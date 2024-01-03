import os
import math
import torch
import numpy as np
import cv2

from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from ezdl.datasets import WeedMapDataset
from cc_torch import connected_components_labeling
from datetime import datetime

from detector import (
    HoughCropRowDetector,
    SplitLawinVegetationDetector,
    ModifiedHoughCropRowDetector,
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
    conn_components = connected_components_labeling(mask).cpu()
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
    outdir,
    checkpoint,
    threshold=150,
    step_theta=1,
    step_rho=1,
    angle_error=3,
    clustering_tol=2,
    uniform_significance=10,
    theta_reduction_threshold=1.00,
):
    summary_file = os.path.join(outdir, "summary.txt")
    os.makedirs(outdir, exist_ok=True)
    with open(summary_file, "w") as f:
        f.write(
            f"checkpoint: {checkpoint}\nthreshold: {threshold}\nstep_theta: {step_theta}\nstep_rho: {step_rho}\nangle_error: {angle_error}\nclustering_tol: {clustering_tol}\nuniform_significance: {uniform_significance}\ntheta_reduction_threshold: {theta_reduction_threshold}\n"
        )
        

def gt_defix(gt):
    gt[gt == 1] = 10000
    return gt


def label(
    root,
    outdir,
    checkpoint,
    threshold=150,
    step_theta=1,
    step_rho=1,
    angle_error=3,
    clustering_tol=2,
    uniform_significance=10,
    theta_reduction_threshold=1.00,
    interactive=False,
):
    outdir = os.path.join(outdir, datetime.now().strftime('%d-%m-%Y_%H:%M:%S'))
    os.makedirs(outdir, exist_ok=True)
    out_summary_file(
        outdir,
        checkpoint,
        threshold=threshold,
        step_theta=step_theta,
        step_rho=step_rho,
        angle_error=angle_error,
        clustering_tol=clustering_tol,
        uniform_significance=uniform_significance,
        theta_reduction_threshold=theta_reduction_threshold,
    )
    channels = ["R", "G", "B", "NIR", "RE"]
    input_transform = lambda x: x
    dataset = WeedMapDataset(
        root=root,
        channels=channels,
        transform=input_transform,
        target_transform=lambda x: x,
        return_path=True,
    )
    labeler = SplitLawinVegetationDetector(checkpoint_path=checkpoint)
    detector = HoughCropRowDetector(
        threshold=threshold,
        step_theta=step_theta,
        step_rho=step_rho,
        angle_error=angle_error,
        clustering_tol=clustering_tol,
        uniform_significance=uniform_significance,
        theta_reduction_threshold=theta_reduction_threshold,
        crop_detector=labeler,
    )
    for i, (img, target, path) in enumerate(tqdm(dataset)):
        mask = labeler(img)
        lines, components = detector.predict_from_mask(mask, return_components=True)
        blank = mask.cpu().numpy().astype(np.uint8)
        line_mask = get_drawn_img(
            torch.zeros_like(torch.tensor(blank)).numpy(), lines, color=(255, 0, 255)
        )
        argmask = mask[0].type(torch.uint8)
        weed_map = label_from_row(argmask, torch.tensor(line_mask).permute(2, 0, 1)[0])
        weed_map = weed_map.argmax(dim=0)
        weed_map = gt_defix(weed_map)
        weed_map = weed_map.cpu().numpy().astype(np.uint8)
        img_out_path = os.path.join(
            outdir, os.path.basename(path["input_name"] + "_GroundTruth_iMap.png")
        )
        cv2.imwrite(img_out_path, weed_map)
        if interactive:
            yield i
