import os.path
import shutil

import click
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from clearml import Dataset
from tqdm import tqdm

from detector import ModifiedHoughCropRowDetector
from data.spring_wheat import SpringWheatDataset, SpringWheatMaskedDataset
from ezdl.utils.grid import make_grid
from ezdl.utils.utilities import load_yaml

from labeling import label as label_fn

DATA_ROOT = "dataset/processed"
CROP_ROWS_PATH = "dataset/crop_rows"
OUTDIR = "dataset/masks"


@click.group()
def main():
    pass


def get_line_boxes(theta, rho, is_deg, img_width, img_height):
    if is_deg:
        theta = np.deg2rad(theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = (a * rho) + img_width / 2
    y0 = (b * rho) + img_height / 2
    x1 = int(x0 + img_width * (-b))
    y1 = int(y0 + img_height * (a))
    x2 = int(x0 - img_width * (-b))
    y2 = int(y0 - img_height * (a))
    return (x1, y1), (x2, y2)


def get_square_from_lines(img_array, theta, rho, displacement, width, height):
    p1, p2 = get_line_boxes(theta, rho + displacement, True, width, height)
    p3, p4 = get_line_boxes(theta, rho - displacement, True, width, height)
    rect = cv2.minAreaRect(np.array([p1, p2, p3, p4]))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img_array, [box], 0, 255, -1)
    return img_array


@main.command("detect")
@click.option("--inpath", default=DATA_ROOT, type=click.STRING)
@click.option("--mask_outpath", default=CROP_ROWS_PATH, type=click.STRING)
@click.option("--uri", default=None)
@click.option("--hough_threshold", default=10)
@click.option("--angle_error", default=3)
@click.option("--clustering_tol", type=click.STRING, default="crop_as_tol")
@click.option("--input_yaml", type=click.STRING)
def cli_row_detection_springwheat(inpath, hough_threshold, mask_outpath, uri, angle_error, clustering_tol, input_yaml):
    """

    Args:
        input_yaml:
        inpath: Base folder of the dataset
        mask_outpath: Folder where to save the masks
        uri: clearml uri for dataset upload
        hough_threshold:
        angle_error:
        clustering_tol:
    """
    clustering_tol = int(clustering_tol) if clustering_tol.isnumeric() else clustering_tol
    if inpath is None or inpath == '' or os.path.exists(inpath) is False:
        inpath = Dataset.get(
            dataset_name="SpringWheatCropMasks",
            dataset_project="SSL"
            ).get_local_copy()
    if input_yaml is not None:
        input_dict = load_yaml(input_yaml)
        input_dict = {**input_dict, **{"uri": [uri], "inpath": [inpath], "mask_outpath": [mask_outpath]}}
        runs = make_grid(input_dict)
    else:
        runs = [{
            "inpath": inpath,
            "hough_threshold":hough_threshold,
            "mask_outpath": mask_outpath,
            "uri": uri,
            "angle_error": angle_error,
            "clustering_tol": clustering_tol
        }]
    print(runs)
    print(f"\n"
          f"Number of runs: {len(runs)}")
    for i, run in enumerate(runs):
        print(f"Run number: {i} on {len(runs)}")
        row_detection_springwheat(**run)


def row_detection_springwheat(inpath, hough_threshold, mask_outpath, uri, angle_error, clustering_tol):

    crd = ModifiedHoughCropRowDetector(crop_detector="None",
                          threshold=hough_threshold,
                          angle_error=angle_error,
                          clustering_tol=clustering_tol)

    shutil.rmtree(mask_outpath, ignore_errors=True)
    os.makedirs(mask_outpath, exist_ok=True)
    mask_suffix = "_mask.png"
    csv_suffix = "_mask.csv"

    dataset = SpringWheatMaskedDataset(root=inpath, return_path=True, return_img=False, transform=None)

    for img, img_path in tqdm(dataset):
        width, height = img.shape[1:]
        fname = os.path.basename(img_path)
        fname, fext = os.path.splitext(fname)
        lines, displacement = crd.predict_from_mask(img, return_mean_crop_size=True)
        mask = np.zeros((width, height), dtype=np.uint8)
        for theta, rho in lines:
            mask = get_square_from_lines(mask, theta, rho, displacement, width, height)
        # Save the lines
        df = pd.DataFrame(lines.cpu(), columns=["theta", "rho"])
        df.to_csv(os.path.join(mask_outpath, fname + csv_suffix))
        # Save the mask
        Image.fromarray(mask).save(os.path.join(mask_outpath, fname + mask_suffix))
    version = f"hough_t={hough_threshold}||angle_err={angle_error}||clust_tol={clustering_tol}"
    manage_clearml_crop_rows(uri, mask_outpath, version)


@main.command("crop_mask")
@click.option("--inpath", default=DATA_ROOT, type=click.STRING)
@click.option("--mask_outpath", default=DATA_ROOT, type=click.STRING)
@click.option("--uri", default=None)
def crop_mask(inpath, mask_outpath, uri):
    """
    :param inpath: Base folder of the dataset
    :param mask_outpath: Folder where to save the masks
    :param uri: clearml uri for dataset upload
    """
    if inpath is None or inpath == '':
        inpath = Dataset.get(
            dataset_name="SpringWheatProcessed",
            dataset_project="SSL"
            ).get_local_copy()

    crd = ModifiedHoughCropRowDetector()

    shutil.rmtree(mask_outpath, ignore_errors=True)
    os.makedirs(mask_outpath, exist_ok=True)
    crop_mask_suffix = "_cropmask.png"

    dataset = SpringWheatDataset(root=inpath, return_path=True, transform=None)

    for img, img_path in tqdm(dataset):
        fname = os.path.basename(img_path)
        fname, fext = os.path.splitext(fname)
        crop_mask = crd.detect_crop(img).cpu().numpy()
        # Save the crop mask
        Image.fromarray(crop_mask).save(os.path.join(mask_outpath, fname + crop_mask_suffix))
    manage_clearml_crop_mask(uri, mask_outpath)


def manage_clearml_crop_rows(uri, outpath, version=None):
    parent = Dataset.get(
        dataset_name="SpringWheatCropMasks",
        dataset_project="SSL"
    )
    dataset = Dataset.create(
        dataset_name="SpringWheatCropRows",
        dataset_project="SSL",
        dataset_version=version,
        parent_datasets=[parent.id]
    )
    dataset.add_files(path=outpath)
    dataset.upload(output_url=uri)
    dataset.finalize()


def manage_clearml_crop_mask(uri, outpath, version=None):
    parent = Dataset.get(
        dataset_name="SpringWheatProcessed",
        dataset_project="SSL"
    )
    dataset = Dataset.create(
        dataset_name="SpringWheatCropMasks",
        dataset_project="SSL",
        dataset_version=version,
        parent_datasets=[parent.id]
    )
    dataset.add_files(path=outpath)
    dataset.upload(output_url=uri)
    dataset.finalize()


@main.command("label")
@click.option("--root", default=DATA_ROOT, type=click.STRING)
@click.option("--outdir", default=OUTDIR, type=click.STRING)
@click.option("--threshold", default=150, type=click.INT)
@click.option("--checkpoint", default=None)
def label(root, outdir, checkpoint, threshold):
    """
    :param root: Base folder of the dataset
    :param threshold: Threshold for the SplitLawinVegetationDetector
    """
    label_fn(root=root, outdir=outdir, checkpoint=checkpoint, threshold=threshold)


if __name__ == '__main__':
    main()
