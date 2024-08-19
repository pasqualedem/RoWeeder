from enum import Enum
import gc
import inspect
import contextlib
from copy import deepcopy

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import get_scheduler as get_transformers_scheduler

from roweeder.data.utils import DataDict
from roweeder.models.utils import LossOutput, ModelOutput
from roweeder.tracker.abstract_tracker import AbstractLogger
from roweeder.utils.logger import get_logger

logger = get_logger(__name__)


def parse_params(params_dict):
    train_params = params_dict.get("train", {})
    dataset_params = params_dict.get("dataset", {})
    model_params = params_dict.get("model", {})
    dataloader_params = params_dict.get("dataloader", {})

    return train_params, dataset_params, dataloader_params, model_params


def cast_model(model: torch.nn.Module, precision=torch.float32):
    if precision == torch.float32:
        return model
    model = model.type(precision)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.float()
    return model


class SchedulerStepMoment(Enum):
    BATCH = "batch"
    EPOCH = "epoch"


def get_scheduler(optimizer, num_training_steps, scheduler_params):
    scheduler_params = deepcopy(scheduler_params)
    scheduler_type = scheduler_params.pop("type")
    if scheduler_type is None:
        logger.warning("No scheduler type specified, using None")
        return None, None
    step_moment = scheduler_params.pop("step_moment", None)
    if step_moment is None:
        raise ValueError("step_moment must be specified, choose (batch, epoch)")
    step_moment = SchedulerStepMoment(step_moment)
    num_warmup_steps = scheduler_params.pop("num_warmup_steps", None)
    return (
        get_transformers_scheduler(
            scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            scheduler_specific_kwargs=scheduler_params,
            num_training_steps=num_training_steps,
        ),
        step_moment,
    )


def get_experiment_tracker(accelerator: Accelerator, params: dict) -> AbstractLogger:
    from roweeder.tracker.wandb_tracker import (
        wandb_experiment as platform_logger,
    )
    return platform_logger(accelerator, params)


def check_nan(model, input_dict, output, loss, step, train_params):
    if not train_params.get("check_nan", False):
        return
    if step % train_params["check_nan"] != 0:
        return
    if torch.isnan(loss) or loss.detach() in [torch.inf, -torch.inf]:
        if (
            train_params["check_nan"] == 1
        ):  # Makes sense only if we are checking every step
            state_dict = {
                "model": model.state_dict(),
                "input_dict": input_dict,
                "loss": loss,
                "step": step,
                "output": output,
            }
            torch.save(state_dict, "nan.pt")
        raise ValueError("NaNs in loss")


def handle_oom(model, input_dict, optimizer, epoch, step):
    logger.warning(f"OOM at step {step}")
    logger.warning(torch.cuda.memory_summary())
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "input_dict": input_dict,
        },
        f"oom_epoch_{epoch}_step_{step}.pt",
    )
    optimizer.zero_grad()
    del input_dict
    gc.collect()
    torch.cuda.empty_cache()


@contextlib.contextmanager
def nosync_accumulation(accumulate=False, accelerator=None, model=None):
    if accumulate:
        with accelerator.no_sync(model):
            yield
    else:
        with contextlib.nullcontext():
            yield


class WrapperModule(torch.nn.Module):
    def __init__(self, model, loss) -> None:
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, input_dict):
        gt = input_dict.target
        model_args = inspect.signature(self.model.forward).parameters
        input_dict = {k: v for k, v in input_dict.items() if k in model_args}
        result_dict = self.model(**input_dict)
        loss = self.loss(result_dict, gt) if self.loss is not None else LossOutput(value=torch.tensor(0.0))
        return ModelOutput(loss=loss, logits=result_dict.logits, scores=result_dict.scores)

    def get_learnable_params(self, train_params):
        model_params = list(self.model.get_learnable_params(train_params))
        loss_params = list(self.loss.parameters())
        if loss_params:
            loss_params = [{"params": loss_params}]
        return model_params + loss_params
    
    @property
    def class_embeddings(self):
        return self.model.class_embeddings

    @class_embeddings.setter
    def class_embeddings(self, value):
        self.model.class_embeddings = value
