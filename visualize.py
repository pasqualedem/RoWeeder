import base64
import contextlib
from io import BytesIO

import streamlit as st
from PIL import Image

from clearml import Dataset, StorageManager, Task
from clearml.backend_config.config import Config
from urllib.parse import urlparse, urlunparse

import os
import numpy as np
import pandas as pd

from utils import remove_suffix


def get_datasets():
    name = "SpringWheatCropRows"

    versions = list(map(lambda x: x['id'], Dataset.list_datasets(partial_name=name)))
    try:
        print("Getting datasets information")
        datasets = list(map(lambda id: Dataset.get(dataset_id=id), versions))
    except ValueError:
        import warnings
        warnings.warn("Url not working, trying for file server")
        config = Config()
        config.reload()
        _, files_server, _, _, _, _ = urlparse(config.get("api").get("files_server"))
        print(f"File server: {files_server}")

        def get_dataset_copy(dataset_id):
            task = Task.get_task(task_id=dataset_id)
            scheme, _, path, params, query, fragment = urlparse(task.artifacts["state"].url)
            url = urlunparse((scheme, files_server, path, params, query, fragment))
            print(f"url: {url}")
            force_download = task.status not in (
                "stopped",
                "published",
                "closed",
                "completed",
            )
            local_state_file = StorageManager.get_local_copy(
                remote_url=url,
                cache_context="datasets",
                extract_archive=False,
                name=task.id,
                force_download=force_download,
            )
            instance = Dataset._deserialize(local_state_file, task)
            return instance

        datasets = list(map(lambda did: get_dataset_copy(did), versions))

    print("Getting local copy of datasets...", end=" ")
    datasets_dict = list(map(lambda x: {'dataset': x, 'path': x.get_local_copy()}, datasets))
    print("Done!")

    files = [x.split('.')[0] for x in os.listdir(datasets_dict[0]['path']) if x.endswith('.JPG')]
    images = list(map(lambda x: x + '.JPG', files))
    crop_masks = list(map(lambda x: x + '_cropmask.png', files))
    crop_rows = list(map(lambda x: x + '_mask.png', files))
    return images, crop_masks, crop_rows, datasets_dict


def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((150, 150), Image.LANCZOS)
    return i


def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


def display_datasets():
    st_state = st.session_state
    i = st_state['i']
    dataset_dict = st.session_state['dataset_dict']
    img_path = dataset_dict[0]['path']
    crop_rows = st_state['rows']

    col1,  col2 = st.columns(2)
    with col1:
        st.write('## Image')
        st.image(Image.open(os.path.join(img_path, st_state['images'][i])), width=300)
    with col2:
        st.write('## Mask')
        st.image(Image.open(os.path.join(img_path, st_state['masks'][i])), width=300)

    st.write('## Crop Rows')
    n_cols = 4
    for j in range(0, len(dataset_dict), n_cols):
        chunk = dataset_dict[j:j + n_cols]
        row = st.columns(len(chunk))
        for k in range(len(chunk)):
            with row[k]:
                st.write(f'{dataset_dict[j + k]["dataset"].version}')
                crop_row = os.path.join(dataset_dict[j + k]['path'], crop_rows[i])
                st.image(Image.open(crop_row), width=300)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    if "images" not in st.session_state:
        images, crop_masks, crop_rows, dataset_dict = get_datasets()
        st.session_state["images"] = images
        st.session_state["masks"] = crop_masks
        st.session_state["dataset_dict"] = dataset_dict
        st.session_state["rows"] = crop_rows

    st.session_state['i'] = st.slider('i', max_value=len(st.session_state['images']))  # 👈 this is a widget

    display_datasets()
