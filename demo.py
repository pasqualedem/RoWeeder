import base64
from io import BytesIO

import streamlit as st
from PIL import Image

from torchmetrics.functional import f1_score
import torchvision.transforms as F

import torch
import numpy as np
import pandas as pd

from roweeder.models import RoWeederFlat
from roweeder.detector import (
    HoughCropRowDetector,
    HoughDetectorDict,
    get_vegetation_detector as get_vegetation_detector_fn,
)
from roweeder.data import get_dataset
from roweeder.labeling import get_drawn_img, label_from_row
from roweeder.visualize import map_grayscale_to_rgb

import lovely_tensors as lt
lt.monkey_patch()


def change_state(src, dest):
    st.session_state[dest] = st.session_state[src]
    st.session_state["i"] = st.session_state[src]


@st.cache_resource
def get_vegetation_detector(ndvi_threshold=0.2):
    name = st.session_state["vegetation_detector"]
    params = {
        "threshold": ndvi_threshold,
    }
    return get_vegetation_detector_fn(name, params)


@st.cache_resource
def get_roweeder():
    model = RoWeederFlat.from_pretrained("pasqualedem/roweeder_flat_512x512").to(
        st.session_state["device"]
    )
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=st.session_state["device"])
    std = torch.tensor([0.229, 0.224, 0.225], device=st.session_state["device"])
    preprocess = F.Compose(
        [
            F.Normalize(mean=mean, std=std),
        ]
    )
    def predict(x):
        x = preprocess(x)
        st.write(x)
        with torch.no_grad():
            out = model(x)
        return out
    return predict


def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((150, 150), Image.LANCZOS)
    return i


def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, "jpeg")
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


def gt_fix(gt):
    gt[gt == 10000] = 1
    gt[gt == 240] = 2
    return gt


def display_prediction():
    st_state = st.session_state
    device = st_state["device"]
    if st_state["img_name"] != "":
        found = False
        for k, name in enumerate(st_state["images"]):
            if st_state["img_name"] in name:
                i = k
                found = True
                st_state["i"] = i
        if not found:
            st.write("img not found")
            return
    else:
        try:
            i = st_state["i"]
        except KeyError:
            i = 0
            st_state["i"] = i

    data_dict = st_state["dataset"][i]
    img = data_dict.image
    gt = data_dict.target

    mask = st.session_state["labeller"](img)
    detector = HoughCropRowDetector(
        threshold=st.session_state["threshold"],
        crop_detector=st.session_state["labeller"],
        step_theta=st.session_state["step_theta"],
        step_rho=st.session_state["step_rho"],
        angle_error=st.session_state["angle_error"],
        clustering_tol=st.session_state["clustering_tol"],
        uniform_significance=st.session_state["uniform_significance"],
        theta_reduction_threshold=st.session_state["theta_reduction_threshold"],
        theta_value=st.session_state["theta_value"],
    )
    res = detector.predict_from_mask(mask)
    lines = res[HoughDetectorDict.LINES]
    original_lines = res[HoughDetectorDict.ORIGINAL_LINES]
    uniform_significance = res[HoughDetectorDict.UNIFORM_SIGNIFICANCE]
    zero_reason = res[HoughDetectorDict.ZERO_REASON]

    gt = gt_fix(torch.tensor(np.array(gt))).cuda()

    to_draw_gt = (gt.cpu().numpy()).astype(np.uint8)
    to_draw_gt = map_grayscale_to_rgb(to_draw_gt)
    to_draw_mask = mask.cpu().numpy().astype(np.uint8)
    line_mask = get_drawn_img(
        torch.zeros_like(torch.tensor(to_draw_mask)).numpy(), lines, color=(255, 0, 255)
    )
    argmask = mask[0].type(torch.uint8)
    weed_map, _, _ = label_from_row(
        img, argmask, torch.tensor(line_mask).permute(2, 0, 1)[0]
    )
    f1 = f1_score(
        weed_map.argmax(dim=0).cuda(),
        gt,
        num_classes=3,
        average="macro",
        task="multiclass",
        multidim_average="global",
    )
    weed_map = weed_map.argmax(dim=0).cpu().numpy().astype(np.uint8)
    weed_map = map_grayscale_to_rgb(
        weed_map, mapping={1: (0, 255, 0), 2: (255, 0, 0)}
    ).transpose(2, 0, 1)
    weed_map_lines = get_drawn_img(weed_map, lines, color=(255, 0, 255))
    to_draw_mask = to_draw_mask[0]
    weed_map = np.moveaxis(weed_map, 0, -1)

    st.write(data_dict.name)
    st.write("f1 score: ", f1)
    st.write("uniform_significance: ", uniform_significance)
    st.write("zero_reason: ", zero_reason)

    col1, col2 = st.columns(2)
    with col1:
        st.write("## Image")
        output_img = (img[:3].squeeze(0).permute(1, 2, 0).numpy() * 255).astype(
            np.uint8
        )
        output_img = Image.fromarray(output_img)
        st.image(output_img, width=300)
    with col2:
        st.write("## Detected plants")
        st.image(Image.fromarray(to_draw_mask), width=300)
    # with col2:
    #     st.write("## GT")
    #     st.image(Image.fromarray(to_draw_gt), width=300)
    col1, col2 = st.columns(2)
    with col1:
        st.write("## Prediction using Lines")
        st.image(weed_map, width=300)
    with col2:
        st.write("## Lines")
        st.image(Image.fromarray(weed_map_lines), width=300)

    roweeder_pred = st_state["roweeder"](img[:3].unsqueeze(0).to(device)).logits
    roweeder_pred = roweeder_pred.argmax(dim=1).cpu().numpy().astype(np.uint8)[0]
    st.write(roweeder_pred.shape)
    roweeder_pred = map_grayscale_to_rgb(
        roweeder_pred, mapping={1: (0, 255, 0), 2: (255, 0, 0)}
    )
    st.write(roweeder_pred.shape)
    st.write("## Prediction using RoWeeder")
    st.image(Image.fromarray(roweeder_pred), width=300)


if __name__ == "__main__":
    st.set_page_config(layout="wide")


    with st.sidebar:
        st.session_state["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        st.session_state["roweeder"] = get_roweeder()
        st.text_input(
            value="dataset/patches/512",
            key="root",
            label="root",
        )
        fields = ["000", "001", "002", "003", "004"]
        dataset = get_dataset(st.session_state["root"], "New Dataset", fields)
        st.session_state["dataset"] = dataset

        st.slider(
            "i",
            max_value=len(st.session_state["dataset"]) - 1,
            step=1,
            key="slider_i",
            on_change=lambda: change_state("slider_i", "number_i"),
        )  # 👈 this is a widget
        st.number_input(
            "i",
            key="number_i",
            value=0,
            on_change=lambda: change_state("number_i", "slider_i"),
        )
        st.text_input(value="", label="img_name", key="img_name")
        st.selectbox(
            "Vegetation Detector",
            ["NDVIDetector"],
            key="vegetation_detector",
        )

        ndvi_threshold = st.slider(
            "ndvi_threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=0.2,
            key="ndvi_threshold",
        )
        st.session_state["labeller"] = get_vegetation_detector(
            ndvi_threshold=ndvi_threshold
        )
    col1, col2 = st.columns(2)
    with col1:
        st.slider(
            "threshold", min_value=0, max_value=255, step=1, value=150, key="threshold"
        )
        st.slider(
            "step_theta", min_value=1, max_value=10, step=1, value=1, key="step_theta"
        )
        st.slider(
            "step_rho", min_value=1, max_value=10, step=1, value=1, key="step_rho"
        )
        st.number_input(
            "Fixed theta value that ovverrides the theta mode calculation",
            key="theta_value",
            value=None,
        )
    with col2:
        st.slider(
            "angle_error", min_value=0, max_value=10, step=1, value=3, key="angle_error"
        )
        st.slider(
            "clustering_tol",
            min_value=0,
            max_value=10,
            step=1,
            value=2,
            key="clustering_tol",
        )
        st.slider(
            "uniform_significance",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=0.1,
            key="uniform_significance",
        )
        st.slider(
            "theta_reduction_threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=1.0,
            key="theta_reduction_threshold",
        )

    display_prediction()
