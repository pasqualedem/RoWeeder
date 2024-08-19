from __future__ import annotations

import copy
import gc
import os
import uuid
import pandas as pd
from typing import Mapping

from roweeder.utils.utils import EasyDict
from roweeder.utils.logger import get_logger
from roweeder.experiment.run import Run
from roweeder.experiment.parallel import ParallelRun
from roweeder.utils.utils import get_timestamp, load_yaml, nested_dict_update, update_collection
from roweeder.utils.grid import linearize, linearized_to_string, make_grid
from roweeder.utils.optuna import Optunizer


logger = get_logger(__name__)


class GridSummary:
    def __init__(
        self,
        total_runs,
        total_runs_excl_grid,
        total_runs_to_run,
        total_runs_excl,
    ):
        self.total_runs = total_runs
        self.total_runs_excl_grid = total_runs_excl_grid
        self.total_runs_to_run = total_runs_to_run
        self.total_runs_excl = total_runs_excl

    def update(self, d):
        self.total_runs = d.get("total_runs") or self.total_runs
        self.total_runs_excl_grid = (
            d.get("total_runs_excl_grid") or self.total_runs_excl_grid
        )
        self.total_runs_to_run = d.get("total_runs_to_run") or self.total_runs_to_run
        self.total_runs_excl = d.get("total_runs_excl") or self.total_runs_to_run


class ExpSettings(EasyDict):
    def __init__(self, *args, **kwargs):
        self.start_from_grid = 0
        self.start_from_run = 0
        self.resume = False
        self.resume_last = False
        self.tracking_dir = ""
        self.excluded_files = ""
        self.name = ""
        self.group = ""
        self.continue_with_errors = True
        self.logger = None
        self.search = "grid"
        self.direction = None
        self.n_trials = None
        self.max_parallel_runs = 1
        self.uuid = None
        self.task = None
        self.test_task = None
        self.timestamp = get_timestamp()
        super().__init__(*args, **kwargs)
        self.tracking_dir = self.tracking_dir or ""

    def update(self, e: ExpSettings, **f):
        if e is None:
            return
        self.start_from_grid = e.start_from_grid or self.start_from_grid
        self.start_from_run = e.start_from_run or self.start_from_run
        self.resume = e.resume or self.resume
        self.resume_last = e.resume_last or self.resume_last
        self.tracking_dir = e.tracking_dir or self.tracking_dir
        self.excluded_files = e.excluded_files or self.excluded_files
        self.group = e.group or self.group
        self.logger = e.logger or self.logger
        self.continue_with_errors = (
            not e.continue_with_errors or self.continue_with_errors
        )
        self.search = e.search or self.search
        self.direction = e.direction or self.direction
        self.n_trials = e.n_trials or self.n_trials
        self.max_parallel_runs = e.max_parallel_runs or self.max_parallel_runs
        self.uuid = e.uuid or self.uuid
        self.task = e.task or self.task
        self.test_task = e.test_task or self.test_task


class Experimenter:
    EXP_FINISH_SEP = "#" * 50 + " FINISHED " + "#" * 50 + "\n"
    EXP_CRASHED_SEP = "|\\" * 50 + "CRASHED" + "|\\" * 50 + "\n"

    def __init__(self):
        self.gs = None
        self.exp_settings = ExpSettings()
        self.grids = None

    def calculate_runs(self, settings):
        base_grid = settings["parameters"]
        other_grids = settings["other_grids"]
        self.exp_settings = ExpSettings(settings["experiment"])
        if track_dir := self.exp_settings["tracking_dir"]:
            os.makedirs(track_dir, exist_ok=True)

        print("\n" + "=" * 100)
        complete_grids = [base_grid]
        if other_grids:
            complete_grids += [
                nested_dict_update(copy.deepcopy(base_grid), other_run)
                for other_run in other_grids
            ]
        logger.info(f"There are {len(complete_grids)} grids")

        if self.exp_settings.search == "grid":
            self.generate_grid_search(complete_grids, other_grids)
        elif self.exp_settings.search == "optim":
            self.generate_optim_search(complete_grids)
        else:
            raise ValueError(f"Unknown search type: {self.exp_settings.search}")
        self.grids = [
            [{"experiment": {**self.exp_settings}, **params} for params in grid]
            for grid in self.grids
        ]
        return self.gs, self.grids, complete_grids

    def generate_optim_search(self, complete_grids):
        fname = f"{self.exp_settings.name}_{self.exp_settings.group.replace('/', '_')}"
        study_names = [f"{fname}_{i}" for i in range(len(complete_grids))]
        self.grids = [
            Optunizer(
                study_name=name,
                grid=grid,
                storage_base=self.exp_settings.tracking_dir,
                n_trials=self.exp_settings.n_trials,
                direction=self.exp_settings.direction,
            )
            for name, grid in zip(study_names, complete_grids)
        ]
        self.generate_grid_summary()

    def generate_grid_search(self, complete_grids, other_grids):
        self.grids, dot_elements = zip(
            *[
                make_grid(grid, return_cartesian_elements=True)
                for grid in complete_grids
            ]
        )
        # WARNING: Grids' objects have the same IDs!
        dot_elements = list(dot_elements)
        if len(dot_elements) > 1:
            dot_elements[1:] = [
                list(dict(linearize(others) + dot).items())
                for others, dot in zip(other_grids, dot_elements[1:])
            ]

        for i, grid in enumerate(self.grids):
            info = f"Found {len(grid)} runs from grid {i}"
            last_grid = (
                self.exp_settings.start_from_grid
                if self.exp_settings.start_from_grid is not None
                else len(self.grids)
            )
            if i < last_grid:
                info += f", skipping grid {i} with {len(grid)} runs"
            logger.info(info)
        self.generate_grid_summary()

        if self.exp_settings.excluded_files:
            os.environ["WANDB_IGNORE_GLOBS"] = self.exp_settings.excluded_files

        print_preview(self, self.gs, self.grids, dot_elements)
        print("=" * 100 + "\n")

        return self.gs, self.grids, dot_elements

    def generate_grid_summary(self):
        total_runs = sum(len(grid) for grid in self.grids)
        if self.exp_settings.start_from_grid is None:
            total_runs_excl_grid = total_runs - len(self.grids[-1])
            total_runs_excl = total_runs
        else:
            total_runs_excl_grid = total_runs - sum(
                len(grid) for grid in self.grids[self.exp_settings.start_from_grid :]
            )
            total_runs_excl = total_runs_excl_grid + self.exp_settings.start_from_run
        total_runs_to_run = total_runs - total_runs_excl
        self.gs = GridSummary(
            total_runs=total_runs,
            total_runs_excl_grid=total_runs_excl_grid,
            total_runs_to_run=total_runs_to_run,
            total_runs_excl=total_runs_excl,
        )

    def execute_runs(self, only_create=False):
        starting_run = self.exp_settings.start_from_run
        for i in range(self.exp_settings.start_from_grid, len(self.grids)):
            grid = self.grids[i]
            if i != self.exp_settings.start_from_grid:
                starting_run = 0
            for j in range(starting_run, len(grid)):
                params = grid[j]
                try:
                    logger.info(f"Running grid {i} out of {len(self.grids) - 1}")
                    logger.info(
                        f"Running run {j} out of {len(grid) - 1} ({sum(len(self.grids[k]) for k in range(i)) + j} / {self.gs.total_runs - 1})"
                    )
                    run = Run()
                    run.init(params)
                    metric = run.launch()
                    print(self.EXP_FINISH_SEP)
                    if self.exp_settings.search == "optim":
                        self.grids[i].report_result(metric)
                    gc.collect()
                except Exception as ex:
                    logger.error(f"Experiment {i} failed with error {ex}")
                    print(self.EXP_CRASHED_SEP)
                    if not self.exp_settings.continue_with_errors:
                        raise ex

    def update_settings(self, d):
        self.exp_settings = update_collection(self.exp_settings, d)
        if self.gs is None:
            return
        self.gs.update(self.exp_settings)
        if "resume" in d:
            self.manage_resume()
            self.generate_grid_summary()


class ParallelExperimenter(Experimenter):
    EXP_FINISH_SEP = "#" * 50 + " LAUNCHED " + "#" * 50 + "\n"
    EXP_CRASHED_SEP = "|\\" * 50 + "CRASHED" + "|\\" * 50 + "\n"

    def __init__(self):
        super().__init__()

    def execute_runs(self, only_create=False):
        starting_run = self.exp_settings.start_from_run
        self.exp_settings.uuid = self.exp_settings.uuid or str(uuid.uuid4())[:8]

        for i in range(self.exp_settings.start_from_grid, len(self.grids)):
            grid = self.grids[i]
            if i != self.exp_settings.start_from_grid:
                starting_run = 0
            for j in range(starting_run, len(grid)):
                params = grid[j]
                try:
                    logger.info(f"Running grid {i} out of {len(self.grids) - 1}")
                    logger.info(
                        f"Running run {j} out of {len(grid) - 1} ({sum(len(self.grids[k]) for k in range(i)) + j} / {self.gs.total_runs - 1})"
                    )
                    run = ParallelRun(
                        experiment_timestamp=self.exp_settings.timestamp,
                        params={"experiment": {**self.exp_settings}, **params},
                    )
                    metric = run.launch(only_create=only_create)
                    print(self.EXP_FINISH_SEP)
                    if self.exp_settings.search == "optim":
                        self.grids[i].report_result(metric)
                    gc.collect()
                except Exception as ex:
                    logger.error(f"Experiment {i} failed with error {ex}")
                    print(self.EXP_CRASHED_SEP)
                    if not self.exp_settings.continue_with_errors:
                        raise ex


def experiment(
    param_path: str = "parameters.yaml",
    parallel: bool = False,
    only_create: bool = False,
    preview: bool = False,
):
    logger.info("Running experiment")
    settings = load_yaml(param_path)
    logger.info(f"Loaded parameters from {param_path}")

    experimenter = ParallelExperimenter() if parallel or only_create else Experimenter()
    experimenter.calculate_runs(settings)
    if not preview:
        experimenter.execute_runs(only_create=only_create)


def run(param_path: str = "parameters.yaml"):
    logger.info("Running run")
    settings = load_yaml(param_path)
    logger.info(f"Loaded parameters from {param_path}")
    single_run = Run()
    single_run.init(settings)
    single_run.launch()
    
    
def test(param_path: str = "parameters.yaml"):
    logger.info("Running run")
    settings = load_yaml(param_path)
    logger.info(f"Loaded parameters from {param_path}")
    if "experiment" in settings: # It's a grid
        experimenter = Experimenter()
        summary, grids, dot_elements = experimenter.calculate_runs(settings)
    else:
        grids = [[settings]]
    for i, grid in enumerate(grids):
        print(f"Grid {i+1} with {len(grid)} runs")
        for j, params in enumerate(grid):
            print(f"Run {j+1} out of {len(grid)}")
            run = Run()
            run.init(params)
            run.test()
            run.end()

def row_test(param_path: str = "parameters.yaml"):
    logger.info("Running run")
    settings = load_yaml(param_path)
    logger.info(f"Loaded parameters from {param_path}")
    if "experiment" in settings: # It's a grid
        experimenter = Experimenter()
        summary, grids, dot_elements = experimenter.calculate_runs(settings)
    else:
        grids = [[settings]]
    for i, grid in enumerate(grids):
        print(f"Grid {i+1} with {len(grid)} runs")
        for j, params in enumerate(grid):
            print(f"Run {j+1} out of {len(grid)}")
            run = Run()
            run.init(params)
            run.row_test()
            run.end()
            

def measure(param_path: str = "parameters.yaml"):
    logger.info("Measuring model")
    settings = load_yaml(param_path)
    logger.info(f"Loaded parameters from {param_path}")
    if "experiment" in settings: # It's a grid
        experimenter = Experimenter()
        summary, grids, dot_elements = experimenter.calculate_runs(settings)
    else:
        grids = [[settings]]
    for i, grid in enumerate(grids):
        print(f"Grid {i+1} with {len(grid)} runs")
        for j, params in enumerate(grid):
            print(f"Run {j+1} out of {len(grid)}")
            run = Run()
            run.init(params)
            run.measure()
            run.end()


def preview(settings: Mapping, param_path: str = "local variable"):
    print(f"Loaded parameters from {param_path}")

    experimenter = Experimenter()
    _, _, _ = experimenter.calculate_runs(settings)


def print_preview(experimenter, grid_summary, grids, cartesian_elements):
    summary_series = pd.concat(
        [pd.Series(grid_summary), pd.Series(experimenter.exp_settings.__dict__)]
    )
    summary_string = f"\n{summary_series.to_string()}\n"

    dfs = [
        pd.DataFrame(
            linearized_to_string(dot_element),
            columns=[f"Grid {i}", f"N. runs: {len(grid)}"],
        )
        for i, (dot_element, grid) in enumerate(zip(cartesian_elements, grids))
    ]
    mark_grids = "\n\n".join(df.to_string(index=False) for df in dfs)
    mark_grids = "Most important parameters for each grid \n" + mark_grids
    logger.info(f"\n{summary_string}\n{mark_grids}")
