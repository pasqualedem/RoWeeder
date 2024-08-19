import os.path
import shutil

import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from roweeder.utils.grid import make_grid
from roweeder.utils.utils import load_yaml

from roweeder.data.spring_wheat import SpringWheatDataset, SpringWheatMaskedDataset
from roweeder.labeling import label as label_fn, load_and_label
from roweeder.detector import ModifiedHoughCropRowDetector
from roweeder.utils.utils import get_square_from_lines
from roweeder.preprocess import divide_ortho_into_patches, rotate_ortho

DATA_ROOT = "dataset/processed"
CROP_ROWS_PATH = "dataset/crop_rows"
OUTDIR = "dataset/generated"
PARAMETERS = "parameters.yaml"


@click.group()
def main():
    pass


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


@main.command("experiment")
@click.option(
    "--parameters", default="parameters.yaml", help="Path to the parameters file"
)
@click.option(
    "--parallel", default=False, help="Run the experiments in parallel", is_flag=True
)
@click.option(
    "--only-create",
    default=False,
    help="Creates params files with running them",
    is_flag=True,
)
def experiment(parameters, parallel, only_create):
    from roweeder.experiment.experiment import experiment as run_experiment
    run_experiment(param_path=parameters, parallel=parallel, only_create=only_create)

        
@main.command("run")
@click.option(
    "--parameters", default="parameters.yaml", help="Path to the parameters file"
)
def run(parameters):
    from roweeder.experiment.experiment import run as run_single
    run_single(param_path=parameters)
    
@main.command("test")
@click.option("--parameters", default="parameters.yaml", help="Path to the parameters file")
def test(parameters):
    from roweeder.experiment.experiment import test as run_test
    run_test(param_path=parameters)
    
@main.command("row_test")
@click.option("--parameters", default="parameters.yaml", help="Path to the parameters file")
def test(parameters):
    from roweeder.experiment.experiment import row_test
    row_test(param_path=parameters)
    
    
@main.command("measure")
@click.option("--parameters", default="parameters.yaml", help="Path to the parameters file")
def test(parameters):
    from roweeder.experiment.experiment import measure as run_measure
    run_measure(param_path=parameters)

    
@main.command("rotate")
@click.option("--root", default=DATA_ROOT, type=click.STRING)
@click.option("--outdir", default=OUTDIR, type=click.STRING)
@click.option("--angle", default=150, type=click.INT)
def rotate(root, outdir, angle):
    """
    :param root: Base folder of the dataset
    :param angle: Angle of rotation
    """
    rotate_ortho(input_folder=root, output_folder=outdir, angle=angle)
    
    
@main.command("patchify")
@click.option("--root", default=DATA_ROOT, type=click.STRING)
@click.option("--outdir", default=OUTDIR, type=click.STRING)
@click.option("--patch_size", default=1024, type=click.INT)
def rotate(root, outdir, patch_size):
    """
    :param root: Base folder of the dataset
    :param angle: Angle of rotation
    """
    divide_ortho_into_patches(root, outdir, patch_size)
    

@main.command("label")
@click.option("--outdir", default=OUTDIR, type=click.STRING)
@click.option("--parameters", default=PARAMETERS, type=click.STRING)
def label(outdir, parameters):
    """
    :param outdir: Output directory
    :param parameters: Parameters file
    """
    load_and_label(outdir, param_file=parameters)


if __name__ == '__main__':
    main()
