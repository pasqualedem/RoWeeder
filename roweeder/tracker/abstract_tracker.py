import os
import time
from PIL import Image
from accelerate import Accelerator
import torch

from roweeder.data.utils import DataDict
from roweeder.utils.logger import get_logger

logger = get_logger(__name__)


def main_process_only(func):
    def wrapper(instance, *args, **kwargs):
        accelerator = instance.accelerator
        if accelerator.is_local_main_process:
            return func(instance, *args, **kwargs)

    return wrapper


class AbstractLogger:
    def __init__(
        self,
        experiment,
        accelerator: Accelerator,
        task: str,
        test_task: str = None,
        log_frequency: int = 100,
        train_image_log_frequency: int = 1000,
        val_image_log_frequency: int = 1000,
        test_image_log_frequency: int = 1000,
        experiment_save_delta: int = None,
    ):
        self.experiment = experiment
        self.accelerator = accelerator
        self.local_dir = experiment.dir if hasattr(experiment, "dir") else None
        self.log_frequency = log_frequency
        self.prefix_frequency_dict = {
            "train": train_image_log_frequency,
            "val": val_image_log_frequency,
            "test": test_image_log_frequency,
        }
        self.task = task
        self.test_task = test_task or task
        self.start_time = time.time()
        self.experiment_save_delta = experiment_save_delta

    def _get_class_ids(self, classes):
        res_classes = []
        for c in classes:
            res_classes.append(sorted(list(set(sum(c, [])))))
        return res_classes

    def log_image(
        self, name: str, image_data: Image, annotations=None, metadata=None, step=None
    ):
        raise NotImplementedError

    def add_tags(self, tags):
        raise NotImplementedError
        
    def log_parameters(self, params):
        raise NotImplementedError
    
    def log_metric(self, name, metric, epoch=None):
        raise NotImplementedError

    def log_metrics(self, metrics, epoch=None):
        raise NotImplementedError

    def log_parameter(self, name, parameter):
        raise NotImplementedError

    def log_training_state(self, epoch, subfolder):
        logger.info("Waiting for all processes to finish for saving training state")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_local_main_process:
            subdir = os.path.join(self.local_dir, subfolder)
            self.accelerator.save_state(output_dir=subdir)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_local_main_process:
            self.log_asset_folder(subdir, step=epoch, base_path=self.local_dir)

    def save_experiment_timed(self):
        """
        Save the experiment every `self.time_delta` seconds
        """
        pass

    def save_experiment(self):
        pass
    
    def log_asset_folder(self, path):
        raise NotImplementedError
    
    def log_prediction(
        self,
        batch_idx: int,
        images: torch.tensor,
        gt: torch.Tensor,
        pred: torch.Tensor,
        id2class: dict,
        phase: str,
    ):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def validate(self):
        raise NotImplementedError
        
    def test(self):
        raise NotImplementedError

    def end(self):
        raise NotImplementedError
    
    @property
    def name(self):
        if self.accelerator.is_local_main_process:
            return self.experiment.name
        return None

    @property
    def url(self):
        if self.accelerator.is_local_main_process:
            return self.experiment.url
        return None


"""
Example usage:
==============================================================
image:
- a path (string) to an image
- file-like containg image
- numpy matrix
- tensorflow tensor
- pytorch tensor
- list of tuple of values
- PIL image
$ logger.log_image("image_name", image)
==============================================================
"""
