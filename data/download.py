import os
import zipfile
from pathlib import Path

import wget
import shutil
import click
from clearml import Dataset


DATA_ROOT = 'dataset/raw'

FIELD_A_URL = "https://www.ipb.uni-bonn.de/html/projects/uav_sugarbeets_2015-16/iros-ral-2018-data/Field_A.zip"
FIELD_B_URL = "https://www.ipb.uni-bonn.de/html/projects/uav_sugarbeets_2015-16/iros-ral-2018-data/Field_B.zip"

FIELD_A_ZIP = 'FieldA.zip'
FIELD_B_ZIP = 'FieldB.zip'


def progress_bar(current, total, width=80):
    print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total))


@click.command()
@click.option("--uri", default=None, type=click.STRING)
@click.option("--outpath", default=DATA_ROOT, type=click.STRING)
def download(uri: str = None, outpath: str = DATA_ROOT):
    os.makedirs(outpath, exist_ok=True)
    wget.download(FIELD_A_URL, FIELD_A_ZIP, progress_bar)
    wget.download(FIELD_B_URL, FIELD_B_ZIP, progress_bar)
    with zipfile.ZipFile(FIELD_A_ZIP, 'r') as zip_ref:
        zip_ref.extractall(outpath)
    with zipfile.ZipFile(FIELD_B_ZIP, 'r') as zip_ref:
        zip_ref.extractall(outpath)
    os.remove(FIELD_A_ZIP)
    os.remove(FIELD_B_ZIP)
    manage_clearml(uri, outpath)


def manage_clearml(uri, outpath):
    dataset = Dataset.create(
        dataset_name="SpringWheat",
        dataset_project="SSL",
    )
    dataset.add_files(path=outpath)
    dataset.upload(output_url=uri)
    dataset.finalize()


if __name__ == '__main__':
    download()