import os

import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
import click

from PIL import Image
from clearml import Dataset
from einops import rearrange
from tqdm import tqdm

SIZE = 1024, 1024
DATA_INPATH = "dataset/raw"
DATA_OUTPATH = "dataset/processed"


@click.command()
@click.option("--uri", default=None, type=click.STRING)
@click.option("--inpath", default=DATA_INPATH, type=click.STRING)
@click.option("--outpath", default=DATA_OUTPATH, type=click.STRING)
@click.option("--size", default=SIZE, type=click.Tuple([int, int]))
def preprocess(inpath: str, outpath: str, size: tuple, uri: str = None):
    """
    :param uri: clearml uri for dataset upload
    :param inpath: Base folder of the dataset
    :param outpath: Folder where to save the preprocessed dataset
    :param size: Size of the resulting images
    """
    if inpath is None or inpath == '':
        inpath = Dataset.get(
            dataset_name="SpringWheat",
            dataset_project="SSL",
            dataset_version="raw"
            ).get_local_copy()
    os.makedirs(outpath, exist_ok=True)

    fields = os.listdir(inpath)
    fields.sort()
    field_a, field_b = fields
    fa_folders = [os.path.join(inpath, field_a, sub) for sub in os.listdir(os.path.join(inpath, field_a))]
    fb_folders = [os.path.join(inpath, field_b, sub, "undistorted") for sub in os.listdir(os.path.join(inpath, field_b))]

    images = [os.path.join(folder, img) for folder in fa_folders + fb_folders for img in os.listdir(folder)]
    for image in tqdm(images):
        generate_and_save_windows(image, outpath, size)
    manage_clearml(uri, outpath)


def generate_and_save_windows(image_path, outpath, size):
    img = Image.open(image_path)
    windows = generate_windows(img, size)
    for i in range(windows.shape[0]):
        image_name, image_ext = os.path.basename(image_path).split(".")
        Image.fromarray(windows[i].numpy()).save(os.path.join(outpath, f"{image_name}_{i}.{image_ext}"))


def generate_windows(img: PIL.Image.Image, size):
    img = torch.tensor(np.array(img))
    img = torch.unsqueeze(img, 0)
    img = rearrange(img, "b w h c -> b c w h")
    channels = img.shape[1]
    img = img.to(torch.float32)
    windows = F.unfold(img, size, stride=size)
    windows = windows.to(torch.uint8)
    windows = rearrange(windows, "b (c w h) n -> (b n) c w h", w=size[0], h=size[1], c=channels)
    windows = rearrange(windows, "b c w h -> b w h c")
    return windows


def manage_clearml(uri, outpath):
    dataset = Dataset.create(
        dataset_name="SpringWheatProcessed",
        dataset_project="SSL",
        dataset_version="processed",
        parent_datasets=['SpringWheat']
    )
    dataset.add_files(path=outpath)
    dataset.upload(output_url=uri)
    dataset.finalize()


if __name__ == '__main__':
    preprocess()
