import subprocess
import sys
import uuid
import os

from colorlog import getLogger

from roweeder.utils.utils import write_yaml


logger = getLogger(__name__)

class ParallelRun:
    slurm_command = "sbatch"
    slurm_script = "slurm/launch_run"
    slurm_script_first_parameter = "--parameters="
    slurm_outfolder = "out"
    out_extension = "out"
    param_extension = "yaml"
    slurm_stderr = "-e"
    slurm_stdout = "-o"

    def __init__(self, params: dict, experiment_timestamp: str):
        self.params = params
        self.exp_timestamp = experiment_timestamp
        if "." not in sys.path:
            sys.path.extend(".")

    def launch(self, only_create=False):
        subfolder = f"{self.exp_timestamp}_{self.params['experiment']['group']}"
        out_folder = os.path.join(self.slurm_outfolder, subfolder)
        os.makedirs(out_folder, exist_ok=True)
        
        run_uuid = str(uuid.uuid4())[:8]
        out_file = f"{run_uuid}.{self.out_extension}"
        out_file = os.path.join(out_folder, out_file)
        param_file = f"{run_uuid}.{self.param_extension}"
        param_file = os.path.join(out_folder, param_file)
        write_yaml(self.params, param_file)
        command = [
            self.slurm_command,
            self.slurm_stdout,
            out_file,
            self.slurm_stderr,
            out_file,
            self.slurm_script,
            self.slurm_script_first_parameter + param_file,
        ]
        if only_create:
            logger.info(f"Creating command: {' '.join(command)}")
        else:
            logger.info(f"Launching command: {' '.join(command)}")
            subprocess.run(command)
