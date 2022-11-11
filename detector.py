import cv2
import numpy as np
import pandas as pd
import torch
from ezdl.datasets import WeedMapDatasetInterface
from ezdl.models.lawin import Laweed
from torchvision.transforms import Normalize, ToTensor, Compose
from torch.nn import functional as F
from histogramdd import histogramdd

MODEL_PATH = "Laweed-1v1r4nzz_latest.pth"
TRAIN_FOLDERS = ['000', '001', '002', '004']
CHANNELS = ['R', 'G', 'B']
SUBSET = 'rededge'


def get_circular_interval(inf, sup, interval_max):
    if sup < inf:
        return (0, sup),  (inf, interval_max)
    if sup > interval_max:
        return (0, sup - interval_max), (inf, interval_max)
    return (inf, sup),


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

        self.mean_crop_size = None
        self.diag_len = None

    def hough(self, shape, connection_dataframe):
        width, height = shape

        d = np.sqrt(np.square(height) + np.square(width))
        thetas = np.arange(0, 180, step=self.step_theta)
        rhos = np.arange(-d, d, step=self.step_rho)
        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))

        # Hough accumulator array of theta vs rho
        accumulator = np.zeros((len(thetas), len(rhos)))
        for idx, (x, y, bx, by, pwidth, pheight, num_pixels) in connection_dataframe.iterrows():
            displacement = self.displacement(pwidth, pheight)
            point = (y - height / 2, x - width / 2)
            for theta_idx in range(len(thetas)):
                rho = (point[1] * cos_thetas[theta_idx]) + (point[0] * sin_thetas[theta_idx])
                rho_idx = np.argmin(np.abs(rhos - rho))
                accumulator[theta_idx, rho_idx - displacement: rho_idx + displacement + 1] += 1
        theta_vec, rho_vec = np.where(accumulator > self.threshold)
        return pd.DataFrame({'theta': theta_vec * self.step_theta, 'rho': rho_vec * self.step_rho - d})

    def hough_vec(self, shape, connection_dataframe):
        width, height = shape

        d = np.sqrt(np.square(height) + np.square(width))
        self.diag_len = d + 1

        thetas = torch.arange(0, 180, step=self.step_theta)
        rhos = torch.arange(-d, d + 1, step=self.step_rho)
        cos_thetas = torch.cos(torch.deg2rad(thetas))
        sin_thetas = torch.sin(torch.deg2rad(thetas))

        # Retrieve all the points from dataframe
        points = torch.stack([torch.tensor(connection_dataframe['cy'].values - height / 2),
                              torch.tensor(connection_dataframe['cx'].values - width / 2)], dim=1)

        # Calculate the rhos values with dot product
        rho_values = torch.matmul(points.float(), torch.stack([sin_thetas, cos_thetas]))

        # Calculate the displacement for each point to add more lines

        shapes = torch.stack([torch.tensor(connection_dataframe['width'].values),
                              torch.tensor(connection_dataframe['height'].values)], dim=1)
        displacement = shapes.max(dim=1).values // 2

        # Hough accumulator array of theta vs rho
        accumulator, edges = histogramdd(
            thetas, rhos, rho_values,
            displacements=displacement
        )
        return accumulator

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
        conn = pd.DataFrame({"cx": centroids[:, 0], "cy": centroids[:, 1],
                             "bx": stats[:, 0], "by": stats[:, 1], "width": stats[:, 2], "height": stats[:, 3],
                             "num_pixels": num_pixels}).drop(0)  # Remove first
        self.mean_crop_size = (conn['width'].mean() + conn['height'].mean()) / 2
        return conn

    def calculate_mask(self, shape, connection_dataframe):
        displ_mask = np.zeros(shape, dtype=np.uint8)
        for idx, (x, y, bx, by, width, height, num_pixels) in connection_dataframe.iterrows():
            displacement = self.displacement(width, height)
            c1 = int(x - displacement), int(y - displacement)
            c2 = int(x + displacement), int(y + displacement)
            # cv.circle(imgc, (int(x), int(y)), 10, (255, 0, 0), 5)
            cv2.rectangle(displ_mask, c1, c2, 255, -1)
        return displ_mask

    def filter_lines(self, accumulator):
        filtered = F.threshold(accumulator, self.threshold, 0)
        mode = filtered.sum(axis=1).argmax().item()
        intervals = get_circular_interval(mode - self.angle_error, mode + self.angle_error + 1, 180 // self.step_theta)
        filtered = torch.concat([filtered[inf:sup] for inf, sup in intervals])
        theta_index = torch.concat([torch.arange(inf, sup) for inf, sup in intervals])
        return filtered, theta_index

    def positivize_rhos(self, accumulator, theta_index):
        chunk_size = int(self.diag_len)
        negatives, positives = accumulator.split(chunk_size, dim=1)
        theta_index = torch.concat([theta_index, theta_index + 180])
        accumulator = torch.concat([positives, negatives.flip(dims=[1])])
        return accumulator, theta_index

    def cluster_lines(self, acc: torch.Tensor, thetas_idcs):
        tol = 2
        thetas_rhos = torch.stack(torch.where(acc > 0), dim=1)
        thetas_rhos[:, 0] = thetas_idcs[thetas_rhos[:, 0]]
        cluster_indices = []
        prec_i = 0
        # TODO RHOS ARE NOT ORDERED!!!!!!!!!!
        for i in range(1, thetas_rhos.shape[0]):
            if abs(thetas_rhos[i][1] - thetas_rhos[i - 1][1]) > tol:
                cluster_indices.append(i - prec_i)
                prec_i = i
        cluster_indices.append(i - prec_i + 1)
        clustered = thetas_rhos.split(cluster_indices)
        return clustered

    def get_medians(self, clusters):
        medians = [cluster[len(cluster)//2] for cluster in clusters]
        return medians

    def predict(self, input_img):
        width, height = input_img.shape[2:]
        crop_mask = self.detect_crop(input_img)
        connectivity_df = self.calculate_connectivity(crop_mask.squeeze(0).cpu().numpy().astype(np.uint8))
        enhanced_mask = self.calculate_mask((width, height), connectivity_df)
        accumulator = self.hough_vec(enhanced_mask.shape, connectivity_df)
        filtered_acc, theta_index = self.filter_lines(accumulator)
        pos_acc, theta_index = self.positivize_rhos(filtered_acc, theta_index)
        clusters = self.cluster_lines(pos_acc, theta_index)
        median_theta, median_rhos = self.get_median_theta_rhos(pos_acc, theta_index, clusters)
        return median_theta, median_rhos
