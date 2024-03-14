import base64
import contextlib
from io import BytesIO

import streamlit as st
from PIL import Image

from urllib.parse import urlparse, urlunparse

import os
import numpy as np
import pandas as pd


def change_state(src, dest):
    st.session_state[dest] = st.session_state[src]
    st.session_state['i'] = st.session_state[src]


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
    if st_state['img_name'] != "":
        found = False
        for k, name in enumerate(st_state['images']):
            if st_state['img_name'] in name:
                i = k
                found = True
                st_state['i'] = i
        if not found:
            st.write("img not found")
            return
    else:
        i = st_state['i']
    dataset_dict = st.session_state['dataset_dict']
    img_path = dataset_dict[0]['path']
    crop_rows = st_state['rows']

    col1,  col2 = st.columns(2)
    st.write(st_state['images'][i])
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

    st.slider('i', max_value=len(st.session_state['images']), step=1, key="slider_i", on_change=lambda: change_state("slider_i", "number_i"))  # 👈 this is a widget
    st.number_input('i', key='number_i', value=0, on_change=lambda: change_state("number_i", "slider_i"))
    st.text_input(value="", label="img_name", key="img_name")

    display_datasets()


def map_grayscale_to_rgb(img, mapping=None):
    if mapping is None:
        mapping = {
            1: (0, 255, 0),  # Value 240 maps to (255, 255, 0) in RGB
            2: (255, 0, 0),  # Value 254 maps to (255, 0, 0) in RGB
        }
    # Initialize an empty RGB image
    rgb_image = np.zeros((*img.shape, 3), dtype=np.uint8)

    # Map greyscale values to RGB using the defined mapping
    for greyscale_value, rgb_color in mapping.items():
        mask = img == greyscale_value
        rgb_image[mask] = rgb_color
    return rgb_image
