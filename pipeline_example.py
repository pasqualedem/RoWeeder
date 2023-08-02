import os

from clearml import PipelineController, Dataset
from PIL import Image
import torch
import os


# We will use the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
def step_one(max_value):
    path = "test/stage1"
    os.makedirs(path, exist_ok=True)
    for i in range(100):
        x = torch.rand(256, 256) * max_value
        x = x.type(torch.uint8)
        Image.fromarray(x.numpy()).save(f"test/stage1/img{i}.png")

    dataset = Dataset.create(
        dataset_name="SpringWheatCropMasks",
        dataset_project="pipex",
        dataset_version=max_value
    )
    dataset.add_files(path=path)
    dataset.upload()
    dataset.finalize()
    dataset_id = dataset.id
    return dataset_id, path


# We will use the following function an independent pipeline component step
# notice all package imports inside the function will be automatically logged as
# required packages for the pipeline execution step
def step_two(dataset_id, path):
    if path is None:
        path = Dataset.get(dataset_id=dataset_id).get_local_copy()
    return len(os.listdir(path))


if __name__ == '__main__':
    # create the pipeline controller
    pipe = PipelineController(
        project='pipex',
        name='Pipeline demo',
        version='1.1',
        add_pipeline_tags=False,
    )

    # set the default execution queue to be used (per step we can override the execution)
    pipe.set_default_execution_queue('default')

    # add pipeline components
    pipe.add_parameter(
        name='max_value',
        description='max value',
        default=256
    )
    pipe.add_function_step(
        name='step_one',
        function=step_one,
        function_kwargs=dict(max_value='${pipeline.max_value}'),
        function_return=['dataset_id', 'path'],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name='step_two',
        # parents=['step_one'],  # the pipeline will automatically detect the dependencies based on the kwargs inputs
        function=step_two,
        function_kwargs=dict(dataset_id='${step_one.dataset_id}', path='$step_one.path'),
        function_return=['len'],
        cache_executed_step=True,
    )

    # For debugging purposes run on the pipeline on current machine
    # Use run_pipeline_steps_locally=True to further execute the pipeline component Tasks as subprocesses.
    # pipe.start_locally(run_pipeline_steps_locally=False)

    # Start the pipeline on the services queue (remote machine, default on the clearml-server)
    pipe.start_locally()

    print('pipeline completed')
