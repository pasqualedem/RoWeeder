import contextlib
import os
import sys
import shutil
from copy import deepcopy
from ptflops import get_model_complexity_info
from safetensors import safe_open

import torch

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.optim import AdamW
from torchmetrics import MetricCollection, F1Score
from tqdm import tqdm

from roweeder.detector import HoughCropRowDetector, HoughDetectorDict, get_vegetation_detector
from roweeder.labeling import get_line_mask, get_on_off_row_plants
from roweeder.models.utils import LossOutput, ModelOutput
from roweeder.utils.logger import get_logger
from roweeder.data import get_dataloaders
from roweeder.data.utils import DataDict
from roweeder.experiment.utils import WrapperModule
from roweeder.loss import build_loss
from roweeder.models import build_model
from roweeder.utils.metrics import RowF1Score, build_metrics
from roweeder.utils.utils import (
    RunningAverage,
    write_yaml,
)

from .utils import (
    SchedulerStepMoment,
    check_nan,
    get_experiment_tracker,
    get_scheduler,
    handle_oom,
    parse_params,
)
from copy import deepcopy

logger = get_logger(__name__)


class Run:
    def __init__(self):
        self.params = None
        self.dataset = None
        self.experiment = None
        self.tracker = None
        self.dataset_params = None
        self.train_params = None
        self.model = None
        self.scheduler = None
        self.criterion = None
        self.best_metric = None
        self.scheduler_step_moment = None
        self.watch_metric = None
        self.train_metrics: MetricCollection = None
        self.val_metrics: MetricCollection = None
        if "." not in sys.path:
            sys.path.extend(".")
        self.global_train_step = 0
        self.global_val_step = 0
        self.validation_json = None
        self.task = None
        self.test_task = None

    def parse_params(self, params: dict):
        self.params = deepcopy(params)

        (
            self.train_params,
            self.dataset_params,
            self.dataloader_params,
            self.model_params,
        ) = parse_params(self.params)

    def init(self, params: dict):
        set_seed(params["seed"])
        self.seg_trainer = None
        logger.info("Parameters: ")
        write_yaml(params, file=sys.stdout)
        self.parse_params(params)

        kwargs = [
            DistributedDataParallelKwargs(find_unused_parameters=True),
        ]
        logger.info("Creating Accelerator")
        self.accelerator = Accelerator(
            even_batches=False,
            kwargs_handlers=kwargs,
            split_batches=False,
            mixed_precision=self.train_params.get("precision", None),
        )
        logger.info("Initiliazing tracker...")
        self.task = self.params["experiment"]["task"]
        self.test_task = self.params["experiment"].get("test_task", self.task)
        self.tracker = get_experiment_tracker(self.accelerator, self.params)
        self.url = self.tracker.url
        self.name = self.tracker.name
        self.train_loader, self.val_loader, self.test_loader, self.deprocess = (
            get_dataloaders(
                self.dataset_params,
                self.dataloader_params,
            )
        )
        model_name = self.model_params.get("name")
        logger.info(f"Creating model {model_name}")
        self.model = build_model(params=self.model_params)
        self.greater_is_better = self.train_params.get("greater_is_better", True)
        logger.info("Creating criterion")
        self.criterion = None
        self.model = WrapperModule(self.model, self.criterion)

        if self.train_params.get("compile", False):
            logger.info("Compiling model")
            self.model = torch.compile(self.model)

        logger.info("Preparing model")
        if self.params.get("loss"):
            self.criterion = build_loss(self.params["loss"])
            self.model.loss = self.criterion
        self.model = self.accelerator.prepare(self.model)

        if self.params.get("train"):
            self._prep_for_training()
        if self.val_loader:
            logger.info("Preparing validation dataloader")
            self._prep_for_validation()

        self._load_state()

    def _prep_for_training(self):
        logger.info("Creating optimizer")
        self.watch_metric = self.train_params["watch_metric"]
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            params = self.model.module.get_learnable_params(self.train_params)
        else:
            params = self.model.get_learnable_params(self.train_params)
        self.optimizer = AdamW(
            params,
            lr=self.train_params["initial_lr"],
        )

        if scheduler_params := self.train_params.get("scheduler", None):
            self.scheduler, self.scheduler_step_moment = get_scheduler(
                scheduler_params=scheduler_params,
                optimizer=self.optimizer,
                num_training_steps=self.train_params["max_epochs"]
                * len(self.train_loader),
            )

        logger.info("Preparing optimizer, dataloaders and scheduler")
        self.train_loader, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.train_loader, self.optimizer, self.scheduler
        )
        self._init_metrics(self.params, phase="train")

    def _prep_for_validation(self):
        self.val_loader = self.accelerator.prepare(self.val_loader)
        self._init_metrics(self.params, phase="val")

    def _load_state(self):
        if self.tracker.accelerator_state_dir:
            overwritten = False
            # Merge image_encoder dict with the state dict
            if (
                "checkpoint" in self.model_params
                and self.params["model"]["name"] != "lam_no_vit"
            ):
                if hasattr(self.model, "module"):
                    model = self.model.module.model
                else:
                    model = self.model.model
                shutil.copyfile(
                    self.tracker.accelerator_state_dir + "/pytorch_model.bin",
                    self.tracker.accelerator_state_dir + "/pytorch_model.bin.bak",
                )
                state_dict = torch.load(
                    self.tracker.accelerator_state_dir + "/pytorch_model.bin"
                )
                state_dict = {
                    **{
                        "model.image_encoder." + k: v
                        for k, v in model.image_encoder.state_dict().items()
                    },
                    **state_dict,
                }
                torch.save(
                    state_dict,
                    self.tracker.accelerator_state_dir + "/pytorch_model.bin",
                )
                overwritten = True

            try:
                self.accelerator.load_state(self.tracker.accelerator_state_dir)
                # Ripristinate old state
            finally:
                if (
                    "checkpoint" in self.model_params
                    and self.params["model"]["name"] != "lam_no_vit"
                    and overwritten
                ):
                    shutil.copyfile(
                        self.tracker.accelerator_state_dir + "/pytorch_model.bin.bak",
                        self.tracker.accelerator_state_dir + "/pytorch_model.bin",
                    )
                    os.remove(
                        self.tracker.accelerator_state_dir + "/pytorch_model.bin.bak"
                    )

    def launch(self):
        logger.info("Start training loop...")

        # Train the Model
        with self.tracker.train():
            logger.info(
                f"Running Model Training {self.params.get('experiment').get('name')}"
            )
            for epoch in range(self.train_params["max_epochs"]):
                logger.info(f'Epoch: {epoch}/{self.train_params["max_epochs"]}')
                self.train_epoch(epoch)

                metrics = None
                if (
                    self.val_loader
                    and epoch % self.train_params.get("val_frequency", 1) == 0
                ):
                    with self.tracker.validate():
                        logger.info("Running Model Validation")
                        metrics = self.validate_epoch(epoch)
                        self._scheduler_step(SchedulerStepMoment.EPOCH, metrics)
                self.save_training_state(epoch, metrics)

        # Restore best model
        self.restore_best_model()

        if self.test_loader:
            self.test()
        self.end()

    def _metric_is_better(self, metric):
        if self.best_metric is None:
            return True
        if self.greater_is_better:
            return metric > self.best_metric
        return metric < self.best_metric

    def save_training_state(self, epoch, metrics=None):
        if metrics and self._metric_is_better(metrics[self.watch_metric]):
            logger.info(
                f"Saving best model with metric {metrics[self.watch_metric]} as given that metric is greater than {self.best_metric}"
            )
            self.best_metric = metrics[self.watch_metric]
            self.tracker.log_training_state(epoch=epoch, subfolder="best")
        self.tracker.log_training_state(epoch=epoch, subfolder="latest")

    def _get_lr(self):
        if self.scheduler is None:
            return self.train_params["initial_lr"]
        with contextlib.suppress(NotImplementedError):
            if hasattr(self.scheduler, "get_lr"):
                return self.scheduler.get_lr()[0]
        if hasattr(self.scheduler, "optimizer"):
            return self.scheduler.optimizer.param_groups[0]["lr"]
        return self.scheduler.optimizers[0].param_groups[0]["lr"]

    def _scheduler_step(self, moment, metrics=None):
        if moment != self.scheduler_step_moment or self.scheduler is None:
            return
        if moment == SchedulerStepMoment.BATCH:
            self.scheduler.step()
        elif moment == SchedulerStepMoment.EPOCH:
            self.scheduler.step(metrics[self.watch_metric])

    def _forward(
        self,
        input_dict: DataDict,
        epoch: int,
        batch_idx: int,
    ):
        try:
            outputs = self.model(input_dict)
        except RuntimeError as e:
            if "out of memory" in str(e):
                handle_oom(
                    self.model,
                    input_dict,
                    self.optimizer,
                    epoch,
                    batch_idx,
                )
                return e
            raise e
        return outputs

    def _backward(self, batch_idx, input_dict, outputs: ModelOutput, loss_normalizer):
        loss = outputs.loss.value / loss_normalizer
        self.accelerator.backward(loss)
        check_nan(
            self.model,
            input_dict,
            outputs,
            loss,
            batch_idx,
            self.train_params,
        )
        return loss

    def _init_metrics(self, params, phase="train"):
        metrics = params.get(f"{phase}_metrics", None)
        metrics = build_metrics(metrics)
        setattr(self, f"{phase}_metrics", self.accelerator.prepare(metrics))

    def _update_loss(self, loss: LossOutput):
        loss_value = loss.value.item()
        loss_components = {k: v.item() for k, v in loss.components.items()}
        self.tracker.log_metric("loss", loss_value)
        if len(loss_components) > 1:  # If there are multiple components
            for k, v in loss_components.items():
                self.tracker.log_metric(f"{k}_loss", v)

    def _update_metrics(
        self,
        metrics: MetricCollection,
        preds: torch.tensor,
        gt: torch.tensor,
        tot_steps: int,
    ):
        with self.accelerator.no_sync(model=metrics):
            metrics.update(preds, gt)
            metrics_dict = metrics.compute()
        # if tot_steps % self.tracker.log_frequency == 0:
        #     for metric_name, metric_value in metrics_dict.items():
        #         metric_value = torch.mean(self.accelerator.gather(metric_value))
        #         self.tracker.log_metric(metric_name, metric_value)
        # .item() to all values
        metrics_dict = {k: v.item() for k, v in metrics_dict.items()}
        return metrics_dict

    def _update_val_metrics(
        self,
        preds: torch.tensor,
        gt: torch.tensor,
        tot_steps,
        phase="val",
    ):
        phase_metrics = getattr(self, f"{phase}_metrics")
        self.tracker.log_metric("step", self.global_val_step)
        return self._update_metrics(phase_metrics, preds, gt, tot_steps)

    def _update_train_metrics(
        self,
        preds: torch.tensor,
        gt: torch,
        tot_steps: int,
        step: int,
    ):
        self.tracker.log_metric("step", self.global_train_step)
        return self._update_metrics(self.train_metrics, preds, gt, tot_steps)

    def train_epoch(
        self,
        epoch: int,
    ):
        if epoch > 0:
            set_seed(self.params["seed"] + epoch)
            logger.info(f"Setting seed to {self.params['seed'] + epoch}")
        self.tracker.log_metric("start_epoch", epoch)
        self.model.train()
        self.train_metrics.reset()

        loss_avg = RunningAverage()
        loss_components_avg = {
            k: RunningAverage() for k in self.criterion.components.keys()
        }
        loss_normalizer = 1
        # tqdm stuff
        bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            postfix={"loss": 0},
            desc=f"Train Epoch {epoch}/{self.train_params['max_epochs']-1}",
        )
        metric_values = None

        for tot_steps, (batch_idx, batch_dict) in enumerate(bar):
            batch_dict: DataDict
            self.optimizer.zero_grad()
            result_dict = self._forward(batch_dict, epoch, batch_idx)
            loss = self._backward(batch_idx, batch_dict, result_dict, loss_normalizer)
            outputs = result_dict.logits
            preds = outputs.argmax(dim=1)
            self.optimizer.step()
            self._scheduler_step(SchedulerStepMoment.BATCH)

            loss_avg.update(loss.item())
            for k, v in result_dict.loss.components.items():
                loss_components_avg[k].update(v.item())
            self._update_loss(result_dict.loss)

            metric_values = self._update_train_metrics(
                preds,
                batch_dict.target,
                tot_steps,
                batch_idx,
            )
            bar.set_postfix(
                {
                    **metric_values,
                    "loss": loss.item(),
                    "lr": self._get_lr(),
                    **{k: v.compute() for k, v in loss_components_avg.items()},
                }
            )
            self.global_train_step += 1
            self.tracker.save_experiment_timed()

        logger.info("Waiting for everyone")
        self.accelerator.wait_for_everyone()
        logger.info(f"Finished Epoch {epoch}")
        logger.info("Metrics")
        metric_dict = {
            **self.train_metrics.compute(),
            "avg_loss": loss_avg.compute(),
            **{k: v.compute() for k, v in loss_components_avg.items()},
        }
        for k, v in metric_dict.items():
            logger.info(f"{k}: {v}")

        self.tracker.log_metrics(
            metrics=metric_dict,
            epoch=epoch,
        )

    def validate_epoch(self, epoch):
        return self.evaluate(self.val_loader, epoch=epoch, phase="val")

    def evaluate(self, dataloader, epoch=None, phase="val"):
        phase_metrics = getattr(self, f"{phase}_metrics")
        self.model.eval()
        phase_metrics.reset()

        avg_loss = RunningAverage()

        tot_steps = 0
        desc = f"{phase} Epoch {epoch}" if epoch is not None else f"{phase}"
        bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            postfix={"loss": 0},
            desc=desc,
            disable=not self.accelerator.is_local_main_process,
        )
        self.tracker.create_prediction_sequence(phase)
        with torch.no_grad():
            for batch_idx, batch_dict in bar:
                result_dict: ModelOutput = self.model(batch_dict)
                outputs = result_dict.logits
                preds = outputs.argmax(dim=1)

                metrics_value = self._update_val_metrics(
                    preds, batch_dict.target, tot_steps, phase
                )
                loss = result_dict.loss.value

                avg_loss.update(loss.item())
                bar.set_postfix(
                    {
                        **metrics_value,
                        "loss": loss.item(),
                    }
                )
                self.tracker.log_prediction(
                    batch_idx=batch_idx,
                    images=self.deprocess(batch_dict.image),
                    gt=batch_dict.target,
                    pred=preds,
                    id2classes=dataloader.dataset.id2class,
                    phase=phase,
                )

                self.global_val_step += 1

            metrics_dict = {
                **phase_metrics.compute(),
                "loss": avg_loss.compute(),
            }

            self.tracker.log_metrics(
                metrics=metrics_dict,
                epoch=epoch,
            )
        self.tracker.add_prediction_sequence(phase)
        self.accelerator.wait_for_everyone()

        metrics_value = phase_metrics.compute()
        for k, v in metrics_value.items():
            if epoch is not None:
                logger.info(f"{phase} epoch {epoch} - {k}: {v}")
            else:
                logger.info(f"{phase} - {k}: {v}")
        logger.info(f"{phase} Loss: {avg_loss.compute()}")
        return metrics_dict

    def measure(self, dataloader=None):
        dataloader = dataloader or self.test_loader
        dataloader = self.accelerator.prepare(dataloader)
        WARMUP = 10
        ITERATIONS = 10
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Number of parameters: {n_params}")
        self.tracker.log_metric("n_params", n_params)

        macs, params = get_model_complexity_info(
            self.model.model,
            (3, 512, 512),
            as_strings=True,
            backend="pytorch",
            print_per_layer_stat=True,
            verbose=True,
        )
        logger.info(f"MACs: {macs}")
        logger.info(f"Params: {params}")
        self.tracker.log_metric("macs", macs)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        i = 0
        total_time = 0
        logger.info(f"Number of warmup steps: {WARMUP}")
        logger.info(f"Number of batches to measure: {len(dataloader)}")
        logger.info("Measuring time")
        with torch.no_grad():
            for k in range(ITERATIONS):
                logger.info(f"Iteration {k}")
                for batch_dict in dataloader:
                    if i < WARMUP:
                        _ = self.model(batch_dict)
                        continue
                start_event.record()
                _ = self.model(batch_dict)
                end_event.record()
                # Wait for GPU to finish
                torch.cuda.synchronize()
                # Calculate elapsed time
                elapsed_time = start_event.elapsed_time(end_event)
                total_time += elapsed_time
                i += 1
        self.accelerator.wait_for_everyone()
        average_time = total_time / i
        logger.info(f"Average time: {average_time}")
        self.tracker.log_metric("average_time", average_time)

    def restore_best_model(self):
        filename = f"{self.tracker.local_dir}/best/model.safetensors"
        with safe_open(filename, framework="pt") as f:
            weights = {k: f.get_tensor(k) for k in f.keys()}
        self.model.load_state_dict(weights)

    def test(self):
        self.test_loader = self.accelerator.prepare(self.test_loader)
        if "test_metrics" in self.params:
            self._init_metrics(self.params, phase="test")
        else:
            self.test_metrics = self.val_metrics
        with self.tracker.test():
            self.evaluate(self.test_loader, phase="test")

    def row_test(self):
        dataloader = self.accelerator.prepare(self.test_loader)
        self.model.eval()
        assert (
            dataloader.batch_size == 1 or dataloader.batch_sampler.batch_size == 1
        ), "Batch size must be 1 for row testing"

        row_model_params = self.params["row_model"]
        hough_detector_params = row_model_params["hough_detector_params"]
        plant_detector_params = row_model_params["plant_detector_params"]

        plant_detector = get_vegetation_detector(
            plant_detector_params["name"], plant_detector_params["params"]
        )
        detector = HoughCropRowDetector(
            **hough_detector_params,
            crop_detector=plant_detector,
        )

        rowf1 = RowF1Score()
        rowf1 = self.accelerator.prepare(rowf1)
        
        f1 = F1Score(num_classes=3, task="multiclass", ignore_index=-100, average=None)
        f1 = self.accelerator.prepare(f1)

        phase = "row"
        desc = "Row testing"
        bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            postfix={"loss": 0},
            desc=desc,
            disable=not self.accelerator.is_local_main_process,
        )
        self.tracker.create_prediction_sequence(phase)
        with torch.no_grad():
            for batch_idx, batch_dict in bar:
                img = batch_dict.image
                ndvi = batch_dict.ndvi

                result_dict: ModelOutput = self.model(batch_dict)
                outputs = result_dict.logits
                preds = outputs.argmax(dim=1)

                plant_mask = plant_detector(ndvi=ndvi[0])[0]
                lines = detector.predict_from_mask(plant_mask)[HoughDetectorDict.LINES]
                line_mask = torch.tensor(get_line_mask(lines, plant_mask.shape)).to(img.device)

                on_row_plants, off_row_plants = get_on_off_row_plants(
                    plant_mask, line_mask
                )
                
                f1.update(preds, batch_dict.target)
                rowf1.update(preds, batch_dict.target, on_row_plants, off_row_plants)

                self.global_val_step += 1

        metrics_dict = {
            "IntraRowF1": rowf1["IntraRowF1"].compute(),
            "InterRowF1": rowf1["InterRowF1"].compute(),
            "F1": f1.compute(),
        }
        metrics_dict = {k: {f"{k}_{i}": v[i].item() for i, _ in enumerate(v)} for k, v in metrics_dict.items()}
        metrics_dict = {k: v for d in metrics_dict.values() for k, v in d.items()}

        self.tracker.log_metrics(
            metrics=metrics_dict,
            epoch=None,
        )


        for k, v in metrics_dict.items():
            logger.info(f"{phase} - {k}: {v}")
        return metrics_dict

    def end(self):
        logger.info("Ending run")
        self.tracker.end()
        logger.info("Run ended")
