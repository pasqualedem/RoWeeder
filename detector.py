import cv2
import numpy as np
import pandas as pd
import torch
from ezdl.datasets import WeedMapDatasetInterface
from ezdl.models.lawin import Laweed
from torchvision.transforms import Normalize, ToTensor, Compose

MODEL_PATH = "Laweed-1v1r4nzz_latest.pth"
TRAIN_FOLDERS = ['000', '001', '002', '004']
CHANNELS = ['R', 'G', 'B']
SUBSET = 'rededge'


def in_circular_interval(inf, sup, value, interval_max):
    sup = sup % interval_max
    inf = inf % interval_max
    if sup > inf:
        return (inf < value) & (value < sup)
    elif sup == inf:
        return value == inf
    else:
        return (value < sup) | (value > inf)


def max_displacement(width, height):
    return int(width / 2 if width > height else height / 2)


def mean_displacement(width, height):
    return int((width + height) / 4)


def load_laweed(use_cuda=False):
    pth = torch.load(MODEL_PATH)
    model = Laweed({'output_channels': 3, 'num_classes': 3, 'backbone': 'MiT-B0', 'backbone_pretrained': True})
    weights = {k[7:]: v for k, v in pth['net'].items()}
    model.load_state_dict(weights)
    model.eval()
    if use_cuda:
        model.cuda()

    return model


class CropRowDetector:

    def __init__(self,
                 step_theta=1,
                 step_rho=1,
                 threshold=10,
                 angle_error=3,
                 displacement_function=max_displacement,
                 use_cuda=True):
        self.step_theta = step_theta
        self.step_rho = step_rho
        self.threshold = threshold
        self.crop_detector = load_laweed(use_cuda=use_cuda)
        self.displacement = displacement_function
        self.angle_error = angle_error
        self.use_cuda = use_cuda
        means, stds = WeedMapDatasetInterface.get_mean_std(TRAIN_FOLDERS, CHANNELS, SUBSET)
        self.transform = Normalize(means, stds)

    def hough(self, input_img, connection_dataframe):
        width, height = input_img.shape

        step_rho = 1
        step_theta = 1
        d = np.sqrt(np.square(height) + np.square(width))
        thetas = np.arange(0, 180, step=step_theta)
        rhos = np.arange(-d, d, step=step_rho)
        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))

        # Hough accumulator array of theta vs rho
        accumulator = np.zeros((len(thetas), len(rhos)))
        for idx, (x, y, bx, by, pwidth, pheight, num_pixels) in connection_dataframe.iterrows():
            displacement = self.displacement(pwidth, pheight)
            point = (y - height / 2, x - width / 2)
            for theta_idx in range(len(thetas)):
                rho = (point[1] * cos_thetas[theta_idx]) + (point[0] * sin_thetas[theta_idx])
                theta = thetas[theta_idx]
                rho_idx = np.argmin(np.abs(rhos - rho))
                accumulator[theta, rho_idx - displacement: rho_idx + displacement] += 1
        theta_vec, rho_vec = np.where(accumulator > self.threshold)
        return pd.DataFrame({'theta': theta_vec, 'rho': rho_vec - d})

    def detect_crop(self, input_img):
        if self.use_cuda:
            input_img = input_img.cuda()
        tensor_img = self.transform(input_img)
        seg = self.crop_detector(tensor_img)
        seg_class = seg.argmax(1)
        seg_class[seg_class == 2] = 255
        seg_class[seg_class == 1] = 255
        if self.use_cuda:
            seg_class = seg_class.cpu()
        return seg_class

    def calculate_connectivity(self, input_img):
        connectivity = 4
        (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(input_img, connectivity=connectivity,
                                                                                  ltype=cv2.CV_32S)
        num_pixels = stats[:, 4]
        return pd.DataFrame({"cx": centroids[:, 0], "cy": centroids[:, 1],
                             "bx": stats[:, 0], "by": stats[:, 1], "width": stats[:, 2], "height": stats[:, 3],
                             "num_pixels": num_pixels}).drop(0)  # Remove first

    def calculate_mask(self, shape, connection_dataframe):
        displ_mask = np.zeros(shape, dtype=np.uint8)
        for idx, (x, y, bx, by, width, height, num_pixels) in connection_dataframe.iterrows():
            displacement = self.displacement(width, height)
            c1 = int(x - displacement), int(y - displacement)
            c2 = int(x + displacement), int(y + displacement)
            # cv.circle(imgc, (int(x), int(y)), 10, (255, 0, 0), 5)
            cv2.rectangle(displ_mask, c1, c2, 255, -1)
        return displ_mask

    def filter_lines(self, lines_df):
        mode = lines_df['theta'].mode()[0]
        # theta_rho_reduced = theta_rho
        # theta_rho_reduced = theta_rho[(theta_rho['theta'] > median - 10) & (theta_rho['theta'] < median + 10)]
        # theta_rho_reduced = theta_rho[theta_rho['theta'] == int(median)]
        # theta_rho_reduced = theta_rho[theta_rho['theta'] == int(mode)]
        return lines_df[in_circular_interval(mode - self.angle_error, mode + self.angle_error, lines_df['theta'], 180)]

    def predict(self, input_img):
        width, height = input_img.shape[1:]
        crop_mask = self.detect_crop(input_img)
        connectivity_df = self.calculate_connectivity(crop_mask.squeeze(0).cpu().numpy().astype(np.uint8))
        enhanced_mask = self.calculate_mask((width, height), connectivity_df)
        theta_rho = self.hough(enhanced_mask, connectivity_df)
        lines = self.filter_lines(theta_rho)
        return lines
