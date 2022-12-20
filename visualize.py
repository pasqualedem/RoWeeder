import streamlit as st

from clearml import Dataset
from IPython.display import display, HTML

import os
import numpy as np
import pandas as pd


def get_datasets():
    name = "SpringWheatCropRows"

    versions = map(lambda x: x['id'], Dataset.list_datasets(partial_name=name))
    datasets = list(map(lambda id: Dataset.get(dataset_id=id), versions))
    datasets_dict = list(map(lambda x: {'dataset': x, 'path': x.get_local_copy()}, datasets))

    files = set(map(lambda x: x.split('.')[0].removesuffix('_mask'), os.listdir(datasets_dict[0]['path'])))
    images = list(map(lambda x: x + '.JPG', files))
    masks = list(map(lambda x: x + '_mask.png', files))
    return images, masks, datasets_dict


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
