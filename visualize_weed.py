import base64
import contextlib
from io import BytesIO

import streamlit as st
from PIL import Image

from clearml import Dataset, StorageManager, Task
from clearml.backend_config.config import Config
from torchmetrics.functional import f1_score

import torch
import numpy as np
import pandas as pd

from ezdl.datasets import WeedMapDataset

from detector import HoughCropRowDetector, SplitLawinVegetationDetector, ModifiedHoughCropRowDetector
from utils import remove_suffix
from labeling import get_drawn_img, label_from_row, label




def change_state(src, dest):
    st.session_state[dest] = st.session_state[src]
    st.session_state['i'] = st.session_state[src]
    

def get_dataset():
    channels = ['R', 'G', 'B', 'NIR', 'RE']
    input_transform = lambda x: x
    dataset = WeedMapDataset(root=st.session_state['root'], channels=channels,
                                transform=input_transform, target_transform=lambda x: x,
                                return_path=True)
    return dataset


@st.cache
def get_model(checkpoint):
    return SplitLawinVegetationDetector(checkpoint_path=checkpoint)


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


def gt_fix(gt):
    gt[gt == 10000] = 1
    return gt


def map_grayscale_to_rgb(img, mapping=None):
    if mapping is None:
        mapping = {
            240: (0, 255, 0),  # Value 240 maps to (255, 0, 0) in RGB
            254: (255, 0, 0),  # Value 254 maps to (255, 255, 255) in RGB
        }
    # Initialize an empty RGB image
    rgb_image = np.zeros((256, 256, 3), dtype=np.uint8)

    # Map greyscale values to RGB using the defined mapping
    for greyscale_value, rgb_color in mapping.items():
        mask = img == greyscale_value
        rgb_image[mask] = rgb_color
    return rgb_image


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
        try:
            i = st_state['i']
        except KeyError:
            i = 0
            st_state['i'] = i

    col1,  col2, col3 = st.columns(3)
    img, gt, additional = st_state['dataset'][i]

    mask = st.session_state["labeller"](img)
    detector = HoughCropRowDetector(threshold=st.session_state['threshold'], 
                                crop_detector=st.session_state["labeller"],
                                step_theta=st.session_state['step_theta'],
                                step_rho=st.session_state['step_rho'],
                                angle_error=st.session_state['angle_error'],
                                clustering_tol=st.session_state['clustering_tol'],
                                uniform_significance=st.session_state['uniform_significance'],
                                theta_reduction_threshold=st.session_state['theta_reduction_threshold'])
    lines, original_lines = detector.predict_from_mask(mask, return_original_lines=True)

    to_draw_gt = (np.array(gt) * 255).astype(np.uint8)
    to_draw_gt = map_grayscale_to_rgb(to_draw_gt).transpose(2, 0, 1)   
    line_gt = get_drawn_img(to_draw_gt, lines, color=(255, 0, 255))
    to_draw_mask = mask.cpu().numpy().astype(np.uint8)
    line_mask = get_drawn_img(torch.zeros_like(torch.tensor(to_draw_mask)).numpy(), lines, color=(255, 0, 255))
    argmask = mask[0].type(torch.uint8)
    weed_map = label_from_row(argmask, torch.tensor(line_mask).permute(2, 0, 1)[0])
    gt = gt_fix(torch.tensor(np.array(gt))).cuda()
    f1 = f1_score(weed_map.argmax(dim=0).cuda(), gt, num_classes=3, average='macro', mdmc_average="global")
    weed_map = weed_map.argmax(dim=0).cpu().numpy().astype(np.uint8)
    weed_map = map_grayscale_to_rgb(weed_map,
                                    mapping={
                                        1: (0, 255, 0),
                                        2: (255, 0, 0)
                                        }).transpose(2, 0, 1)
    weed_map = get_drawn_img(weed_map, lines, color=(255, 0, 255))

    st.write(additional['input_name'])
    st.write("f1 score: ", f1)
    with col1:
        st.write('## Image')
        output_img = (img[:3].squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        output_img = Image.fromarray(output_img)
        st.image(output_img, width=300)
    with col2:
        st.write('## GT')
        st.image(Image.fromarray(line_gt), width=300)
    with col3:
        st.write('## Prediction')
        st.image(weed_map, width=300)
    st.dataframe(pd.DataFrame(lines.cpu(), columns=["rho", "theta"]))
    st.dataframe(pd.DataFrame((original_lines.cpu() if original_lines is not None else []), columns=["rho", "theta"]))

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    if "root" not in st.session_state:
        st.session_state["root"] = st.text_input(value="../Datasets/WeedMap/0_rotations_processed_003_test/RedEdge", label="root")
    if "dataset" not in st.session_state:
        dataset = get_dataset()
        st.session_state["dataset"] = dataset

    with st.sidebar:
        st.slider('i', max_value=len(st.session_state['dataset']), step=1, key="slider_i", on_change=lambda: change_state("slider_i", "number_i"))  # 👈 this is a widget
        st.number_input('i', key='number_i', value=0, on_change=lambda: change_state("number_i", "slider_i"))
        st.text_input(value="", label="img_name", key="img_name")
        checkpoint = st.text_input(value="checkpoints/SplitLawin_B1_RedEdge_RGBNIRRE.pth", label="checkpoint")
        st.session_state["labeller"] = get_model(checkpoint)
    col1,  col2 = st.columns(2)
    with col1:
        st.slider('threshold', min_value=0, max_value=255, step=1, value=150, key="threshold")
        st.slider('step_theta', min_value=1, max_value=10, step=1, value=1, key="step_theta")
        st.slider('step_rho', min_value=1, max_value=10, step=1, value=1, key="step_rho")
    with col2:
        st.slider('angle_error', min_value=0, max_value=10, step=1, value=3, key="angle_error")
        st.slider('clustering_tol', min_value=0, max_value=10, step=1, value=2, key="clustering_tol")
        st.slider('uniform_significance', min_value=0.0, max_value=1.0, step=0.01, value=0.1, key="uniform_significance")
        st.slider('theta_reduction_threshold', min_value=0.0, max_value=1.0, step=0.01, value=1.0, key="theta_reduction_threshold")

    display_datasets()
    
    st.text_input(value="dataset/generated", label="out_dir", key="out_dir")
    if st.button("label"):
        bar = st.progress(0)
        for i in label(
            root=st.session_state['root'],
            outdir=st.session_state['out_dir'],
            checkpoint=checkpoint,
            threshold=st.session_state['threshold'],
            step_theta=st.session_state['step_theta'],
            step_rho=st.session_state['step_rho'],
            angle_error=st.session_state['angle_error'],
            clustering_tol=st.session_state['clustering_tol'],
            uniform_significance=st.session_state['uniform_significance'],
            theta_reduction_threshold=st.session_state['theta_reduction_threshold'],
            interactive=True,
        ):
            bar.progress(i / len(st.session_state['dataset']))
            
    

    # st.slider('theta', min_value=0.0, max_value=5.0, step=0.01, value=0.0, key="theta")
    # st.slider('rho', min_value=0, max_value=1000, step=1, value=0, key="rho")
    # blank = np.zeros((3, 300, 300), np.uint8)
    # blank = get_drawn_img(blank, [(st.session_state['rho'], st.session_state['theta'])])
    # st.image(blank, width=300)
    # st.write(np.rad2deg(st.session_state['theta']))
    # st.write(np.sin(2*st.session_state['theta']))
    # st.write((1 - np.abs(np.sin(2*st.session_state['theta']))))
    # st.slider('min_reduce', min_value=0.0, max_value=1.0, step=0.01, value=0.5, key="min_reduce")
    # min_reduce = st.session_state['min_reduce']
    # st.write((1 - np.abs(np.sin(2*st.session_state['min_reduce']))) * (1 - min_reduce) / 1 + min_reduce)