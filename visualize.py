import base64
from io import BytesIO

import streamlit as st
from PIL import Image

from clearml import Dataset
from IPython.display import display, HTML

import os
import numpy as np
import pandas as pd

from utils import remove_suffix


def get_datasets():
    name = "SpringWheatCropRows"

    versions = map(lambda x: x['id'], Dataset.list_datasets(partial_name=name))
    datasets = list(map(lambda id: Dataset.get(dataset_id=id), versions))
    datasets_dict = list(map(lambda x: {'dataset': x, 'path': x.get_local_copy()}, datasets))

    files = set(map(lambda x: remove_suffix(x.split('.')[0], '_mask'), os.listdir(datasets_dict[0]['path'])))
    images = list(map(lambda x: x + '.JPG', files))
    masks = list(map(lambda x: x + '_mask.png', files))
    return images, masks, datasets_dict


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
    i = 0
    df = pd.DataFrame({'version': map(lambda x: x['dataset'].version, st.session_state['datasets']['datasets_dict'])})
    df['mask'] = [os.path.join(st.session_state['datasets']['datasets_dict'][j]['path'], images[i])
                  for j in range(len(st.session_state['datasets']['datasets_dict']))]
    df['mask'] = df['mask'].map(get_thumbnail)
    st.write(df.to_html(formatters={'image': image_formatter}, escape=False))


if __name__ == "__main__":
    if "datasets" not in st.session_state:
        images, masks, dataset_dict = get_datasets()
        st.session_state["datasets"] = {"images": images, "masks": masks, "dataset_dict": dataset_dict}

    dataframe = pd.DataFrame(
        np.random.randn(10, 20),
        columns=('col %d' % i for i in range(20)))
    st.table(dataframe)

    x = st.slider('x')  # 👈 this is a widget
    st.write(x, 'squared is', x * x)
