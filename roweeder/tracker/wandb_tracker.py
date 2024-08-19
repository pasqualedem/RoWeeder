from contextlib import contextmanager
from copy import deepcopy
import math
import os
from typing import Optional, Union, Any

import pandas as pd
import numpy as np

import torch.nn.functional as F

import torch
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from roweeder.data.utils import DataDict
from roweeder.tracker.abstract_tracker import AbstractLogger, main_process_only

from accelerate import Accelerator
from roweeder.utils.utils import log_every_n, write_yaml
from roweeder.utils.logger import get_logger


logger = get_logger(__name__)

WANDB_ID_PREFIX = "wandb_id."
WANDB_INCLUDE_FILE_NAME = ".wandbinclude"


def wandb_experiment(accelerator: Accelerator, params: dict):
    logger_params = deepcopy(params.get("tracker", {}))
    wandb_params = {
        "accelerator": accelerator,
        "project_name": params["experiment"]["name"],
        "group": params["experiment"].get("group", None),
        "task": params["experiment"]["task"],
        "test_task": params["experiment"].get(
            "test_task", params["experiment"]["task"]
        ),
        **logger_params,
    }
    wandb_logger = WandBLogger(**wandb_params)
    wandb_logger.log_parameters(params)
    wandb_logger.add_tags(logger_params.get("tags", ()))

    return wandb_logger


class WandBLogger(AbstractLogger):
    MAX_CLASSES = 100000  # For negative classes

    def __init__(
        self,
        project_name: str,
        resume: bool = False,
        offline_directory: str = None,
        cache_directory: str = None,
        save_checkpoints_remote: bool = True,
        save_tensorboard_remote: bool = True,
        save_logs_remote: bool = True,
        entity: Optional[str] = None,
        api_server: Optional[str] = None,
        save_code: bool = False,
        tags=None,
        run_id=None,
        resume_checkpoint_type: str = "best",
        group=None,
        ignored_files=None,
        **kwargs,
    ):
        """

        :param experiment_name: Used for logging and loading purposes
        :param s3_path: If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally
        :param checkpoint_loaded: if true, then old tensorboard files will *not* be deleted when tb_files_user_prompt=True
        :param max_epochs: the number of epochs planned for this training
        :param tb_files_user_prompt: Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard: Whether to launch a TensorBoard process.
        :param tensorboard_port: Specific port number for the tensorboard to use when launched (when set to None, some free port
                    number will be used
        :param save_checkpoints_remote: Saves checkpoints in s3.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote: Saves log files in s3.
        :param save_code: save current code to wandb
        """
        tracker_resume = "must" if resume else None
        self.resume = tracker_resume
        resume = run_id is not None
        if not tracker_resume and resume:
            if tags is None:
                tags = []
            tags = tags + ["resume", run_id]
        self.accelerator_state_dir = None
        if cache_directory:
            os.makedirs(cache_directory, exist_ok=True)
            os.environ["WANDB_ARTIFACT_LOCATION"] = os.path.join(cache_directory, "artifacts")
            os.makedirs(os.path.join(cache_directory, "artifacts"), exist_ok=True)
            os.environ["WANDB_ARTIFACT_DIR"] = os.path.join(cache_directory, "artifacts")
            os.makedirs(os.path.join(cache_directory, "cache"), exist_ok=True)
            os.environ["WANDB_CACHE_DIR"] = os.path.join(cache_directory, "cache")
            os.makedirs(os.path.join(cache_directory, "config"), exist_ok=True)
            os.environ["WANDB_CONFIG_DIR"] = os.path.join(cache_directory, "config")
            os.makedirs(os.path.join(cache_directory, "data"), exist_ok=True)
            os.environ["WANDB_DATA_DIR"] = os.path.join(cache_directory, "data")
            os.makedirs(os.path.join(cache_directory, "media"), exist_ok=True)
        if ignored_files:
            os.environ["WANDB_IGNORE_GLOBS"] = ignored_files
        if resume:
            self._resume(
                offline_directory, run_id, checkpoint_type=resume_checkpoint_type
            )
        experiment = None
        if kwargs["accelerator"].is_local_main_process:
            experiment = wandb.init(
                project=project_name,
                entity=entity,
                resume=tracker_resume,
                id=run_id if tracker_resume else None,
                tags=tags,
                dir=offline_directory,
                group=group,
            )
            logger.info(f"wandb run id  : {experiment.id}")
            logger.info(f"wandb run name: {experiment.name}")
            logger.info(f"wandb run dir : {experiment.dir}")
            wandb.define_metric("train/step")
            # set all other train/ metrics to use this step
            wandb.define_metric("train/*", step_metric="train/step")

            wandb.define_metric("validate/step")
            # set all other validate/ metrics to use this step
            wandb.define_metric("validate/*", step_metric="validate/step")

        super().__init__(experiment=experiment, **kwargs)
        if save_code:
            self._save_code()

        self.save_checkpoints_wandb = save_checkpoints_remote
        self.save_tensorboard_wandb = save_tensorboard_remote
        self.save_logs_wandb = save_logs_remote
        self.context = ""
        self.sequences = {}

    def _resume(self, offline_directory, run_id, checkpoint_type="latest"):
        if not offline_directory:
            offline_directory = "."
        wandb_dir = os.path.join(offline_directory, "wandb")
        runs = os.listdir(wandb_dir)
        runs = sorted(list(filter(lambda x: run_id in x, runs)))
        if len(runs) == 0:
            raise ValueError(f"Run {run_id} not found in {wandb_dir}")
        if len(runs) > 1:
            logger.warning(f"Multiple runs found for {run_id} in {wandb_dir}")
            for run in runs:
                logger.warning(run)
            logger.warning(f"Using {runs[0]}")
        run = runs[0]
        self.accelerator_state_dir = os.path.join(
            wandb_dir, run, "files", checkpoint_type
        )
        logger.info(f"Resuming from {self.accelerator_state_dir}")

    def _save_code(self):
        """
        Save the current code to wandb.
        If a file named .wandbinclude is avilable in the root dir of the project the settings will be taken from the file.
        Otherwise, all python file in the current working dir (recursively) will be saved.
        File structure: a single relative path or a single type in each line.
        i.e:

        src
        tests
        examples
        *.py
        *.yaml

        The paths and types in the file are the paths and types to be included in code upload to wandb
        """
        base_path, paths, types = self._get_include_paths()

        if len(types) > 0:

            def func(path):
                for p in paths:
                    if path.startswith(p):
                        for t in types:
                            if path.endswith(t):
                                return True
                return False

            include_fn = func
        else:
            include_fn = lambda path: path.endswith(".py")

        if base_path != ".":
            wandb.run.log_code(base_path, include_fn=include_fn)
        else:
            wandb.run.log_code(".", include_fn=include_fn)

    @main_process_only
    def log_parameters(self, config: dict = None):
        wandb.config.update(config, allow_val_change=self.resume)
        tmp = os.path.join(self.local_dir, "config.yaml")
        write_yaml(config, tmp)
        # self.add_file("config.yaml")

    @main_process_only
    def add_tags(self, tags):
        wandb.run.tags = wandb.run.tags + tuple(tags)

    @main_process_only
    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0):
        wandb.log(data={tag: scalar_value}, step=global_step)

    @main_process_only
    def add_scalars(self, tag_scalar_dict: dict, global_step: int = 0):
        for name, value in tag_scalar_dict.items():
            if isinstance(value, dict):
                tag_scalar_dict[name] = value["value"]
        wandb.log(data=tag_scalar_dict, step=global_step)

    @main_process_only
    def add_image(
        self,
        tag: str,
        image: Union[torch.Tensor, np.array, Image.Image],
        data_format="CHW",
        global_step: int = 0,
    ):
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        if image.shape[0] < 5:
            image = image.transpose([1, 2, 0])
        wandb.log(data={tag: wandb.Image(image, caption=tag)}, step=global_step)

    @main_process_only
    def add_images(
        self,
        tag: str,
        images: Union[torch.Tensor, np.array],
        data_format="NCHW",
        global_step: int = 0,
    ):
        wandb_images = []
        for im in images:
            if isinstance(im, torch.Tensor):
                im = im.cpu().detach().numpy()

            if im.shape[0] < 5:
                im = im.transpose([1, 2, 0])
            wandb_images.append(wandb.Image(im))
        wandb.log({tag: wandb_images}, step=global_step)

    @main_process_only
    def add_video(
        self, tag: str, video: Union[torch.Tensor, np.array], global_step: int = 0
    ):
        if video.ndim > 4:
            for index, vid in enumerate(video):
                self.add_video(tag=f"{tag}_{index}", video=vid, global_step=global_step)
        else:
            if isinstance(video, torch.Tensor):
                video = video.cpu().detach().numpy()
            wandb.log({tag: wandb.Video(video, fps=4)}, step=global_step)

    @main_process_only
    def add_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.array],
        bins: str,
        global_step: int = 0,
    ):
        wandb.log({tag: wandb.Histogram(values, num_bins=bins)}, step=global_step)

    @main_process_only
    def add_plot(self, tag: str, values: pd.DataFrame, xtitle, ytitle, classes_marker):
        table = wandb.Table(columns=[classes_marker, xtitle, ytitle], dataframe=values)
        plt = wandb.plot_table(
            tag,
            table,
            {"x": xtitle, "y": ytitle, "class": classes_marker},
            {
                "title": tag,
                "x-axis-title": xtitle,
                "y-axis-title": ytitle,
            },
        )
        wandb.log({tag: plt})

    @main_process_only
    def add_text(self, tag: str, text_string: str, global_step: int = 0):
        wandb.log({tag: text_string}, step=global_step)

    @main_process_only
    def add_figure(self, tag: str, figure: plt.figure, global_step: int = 0):
        wandb.log({tag: figure}, step=global_step)

    @main_process_only
    def add_mask(self, tag: str, image, mask_dict, global_step: int = 0):
        wandb.log({tag: wandb.Image(image, masks=mask_dict)}, step=global_step)

    @main_process_only
    def add_table(self, tag, data, columns, rows):
        if isinstance(data, torch.Tensor):
            data = [[x.item() for x in row] for row in data]
        table = wandb.Table(data=data, rows=rows, columns=columns)
        wandb.log({tag: table})

    @main_process_only
    def end(self):
        wandb.finish()

    @main_process_only
    def add_file(self, file_name: str = None):
        pass
        # wandb.save(
        #     glob_str=os.path.join(self.local_dir, file_name),
        #     base_path=self.local_dir,
        #     policy="now",
        # )

    @main_process_only
    def add_summary(self, metrics: dict):
        wandb.summary.update(metrics)

    @main_process_only
    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        name = f"ckpt_{global_step}.pth" if tag is None else tag
        if not name.endswith(".pth"):
            name += ".pth"

        path = os.path.join(self.local_dir, name)
        torch.save(state_dict, path)

        # if self.save_checkpoints_wandb:
        #     if self.s3_location_available:
        #         self.model_checkpoints_data_interface.save_remote_checkpoints_file(
        #             self.experiment_name, self.local_dir, name
        #         )
        #     wandb.save(glob_str=path, base_path=self.local_dir, policy="now")

    @main_process_only
    def _get_tensorboard_file_name(self):
        try:
            tb_file_path = self.tensorboard_writer.file_writer.event_writer._file_name
        except RuntimeError as e:
            logger.warning("tensorboard file could not be located for ")
            return None

        return tb_file_path

    @main_process_only
    def _get_wandb_id(self):
        for file in os.listdir(self.local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                return file.replace(WANDB_ID_PREFIX, "")

    @main_process_only
    def _set_wandb_id(self, id):
        for file in os.listdir(self.local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                os.remove(os.path.join(self.local_dir, file))

    @main_process_only
    def add(self, tag: str, obj: Any, global_step: int = None):
        pass

    @main_process_only
    def _get_include_paths(self):
        """
        Look for .wandbinclude file in parent dirs and return the list of paths defined in the file.

        file structure is a single relative (i.e. src/) or a single type (i.e *.py)in each line.
        the paths and types in the file are the paths and types to be included in code upload to wandb
        :return: if file exists, return the list of paths and a list of types defined in the file
        """

        wandb_include_file_path = self._search_upwards_for_file(WANDB_INCLUDE_FILE_NAME)
        if wandb_include_file_path is not None:
            with open(wandb_include_file_path) as file:
                lines = file.readlines()

            base_path = os.path.dirname(wandb_include_file_path)
            paths = []
            types = []
            for line in lines:
                line = line.strip().strip("/n")
                if line == "" or line.startswith("#"):
                    continue

                if line.startswith("*."):
                    types.append(line.replace("*", ""))
                else:
                    paths.append(os.path.join(base_path, line))
            return base_path, paths, types

        return ".", [], []

    @staticmethod
    def _search_upwards_for_file(file_name: str):
        """
        Search in the current directory and all directories above it for a file of a particular name.
        :param file_name: file name to look for.
        :return: pathlib.Path, the location of the first file found or None, if none was found
        """

        try:
            cur_dir = os.getcwd()
            while cur_dir != "/":
                if file_name in os.listdir(cur_dir):
                    return os.path.join(cur_dir, file_name)
                else:
                    cur_dir = os.path.dirname(cur_dir)
        except RuntimeError as e:
            return None

        return None

    @main_process_only
    def create_prediction_sequence(self, phase, columns=[]):
        name = f"{phase}_predictions"
        tracker_task = self.task if phase in ["train", "val"] else self.test_task
        if tracker_task == "classification":
            columns = ["Ground Truth", "Prediction"] + columns
        self.create_image_sequence(name, columns)

    @main_process_only
    def add_prediction_sequence(self, phase):
        self.add_image_sequence(f"{phase}_predictions")

    @main_process_only
    def create_image_sequence(self, name, columns=[]):
        self.sequences[name] = wandb.Table(["ID", "Image"] + columns)

    @main_process_only
    def add_image_to_sequence(
        self, sequence_name, name, wandb_image: wandb.Image, metadata=[]
    ):
        self.sequences[sequence_name].add_data(name, wandb_image, *metadata)

    @main_process_only
    def add_image_sequence(self, name):
        wandb.log({f"{self.context}_{name}": self.sequences[name]})
        del self.sequences[name]

    @main_process_only
    def log_prediction(
        self,
        batch_idx: int,
        images: DataDict,
        gt: torch.Tensor,
        pred: torch.Tensor,
        id2classes: dict,
        phase: str,
    ):
        if not log_every_n(batch_idx, self.prefix_frequency_dict[phase]):
            return
        tracker_task = self.task if phase in {"train", "val"} else self.test_task
        for b in range(gt.shape[0]):
            image = images[b].permute(1, 2, 0).detach().cpu().numpy()
            sample_gt = gt[b].detach().cpu().numpy()
            sample_pred = pred[b].detach().cpu().numpy()

            if tracker_task == "segmentation":
                kwargs = dict(
                    masks={
                        "ground_truth": {
                            "mask_data": sample_gt,
                            "class_labels": id2classes,
                        },
                        "prediction": {
                            "mask_data": sample_pred,
                            "class_labels": id2classes,
                        },
                    },
                    classes=[{"id": c, "name": name} for c, name in id2classes.items()],
                )
                metadata = []
            elif tracker_task == "classification":
                kwargs = {}
                metadata = [
                    id2classes[sample_gt.item()],
                    id2classes[sample_pred.item()],
                ]
            else:
                raise ValueError(f"Task {self.task} not supported")

            wandb_image = wandb.Image(
                image,
                **kwargs,
            )

            self.add_image_to_sequence(
                f"{phase}_predictions",
                f"image_{batch_idx}_sample_{b}",
                wandb_image,
                metadata=metadata,
            )

    @main_process_only
    def log_asset_folder(self, folder, base_path=None, step=None):
        files = os.listdir(folder)
        # for file in files:
        #     wandb.save(os.path.join(folder, file), base_path=base_path)

    @main_process_only
    def log_metric(self, name, metric, epoch=None):
        if self.context:
            name = f"{self.context}/{name}"
        wandb.log({name: metric})

    @main_process_only
    def log_metrics(self, metrics: dict, epoch=None):
        if self.context:
            metrics = {f"{self.context}/{k}": v for k, v in metrics.items()}
        wandb.log(metrics)

    def __repr__(self):
        return "WandbLogger"

    @contextmanager
    def train(self):
        # Save the old context and set the new one
        old_context = self.context
        self.context = "train"

        yield self

        # Restore the old one
        self.context = old_context

    @contextmanager
    def validate(self):
        # Save the old context and set the new one
        old_context = self.context
        self.context = "validate"

        yield self

        # Restore the old one
        self.context = old_context

    @contextmanager
    def test(self):
        # Save the old context and set the new one
        old_context = self.context
        self.context = "test"

        yield self

        # Restore the old one
        self.context = old_context
