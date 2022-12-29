import base64
import contextlib
from io import BytesIO

import streamlit as st
from PIL import Image

from clearml import Dataset

import os
import numpy as np
import pandas as pd

from utils import remove_suffix


def get_datasets():
    name = "SpringWheatCropRows"

    versions = map(lambda x: x['id'], Dataset.list_datasets(partial_name=name))
    datasets = list(map(lambda id: Dataset.get(dataset_id=id), versions))
    datasets_dict = list(map(lambda x: {'dataset': x, 'path': x.get_local_copy()}, datasets))

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
