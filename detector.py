import cv2
import numpy as np
import pandas as pd
import torch
from cc_torch import connected_components_labeling
from ezdl.datasets import WeedMapDatasetInterface
from ezdl.models.lawin import Laweed
from torchvision.transforms import Normalize, ToTensor, Compose
from torch.nn import functional as F
from histogramdd import histogramdd
from utils import previous_iterator

MODEL_PATH = "Laweed-1v1r4nzz_latest.pth"
TRAIN_FOLDERS = ['000', '001', '002', '004']
CHANNELS = ['R', 'G', 'B']
SUBSET = 'rededge'


def get_circular_interval(inf, sup, interval_max):
    if sup < inf:
        return (0, sup),  (inf, interval_max)
    if sup > interval_max:
        return (0, sup - interval_max), (inf, interval_max)
    if inf < 0:
        return (0, sup), (interval_max + inf, interval_max)
    return (inf, sup),


def max_displacement(width, height):
    return int(width / 2 if width > height else height / 2)


def mean_displacement(width, height):
    return int((width + height) / 4)


def load_laweed():
    pth = torch.load(MODEL_PATH)
    model = Laweed({'output_channels': 3, 'num_classes': 3, 'backbone': 'MiT-B0', 'backbone_pretrained': True})
    weights = {k[7:]: v for k, v in pth['net'].items()}
    model.load_state_dict(weights)
    model.eval()
    model.cuda()

    return model


class CropRowDetector:
    IDX_CX = 0
    IDX_CY = 1
    IDX_X0 = 2
    IDX_Y0 = 3
    IDX_X1 = 4
    IDX_Y1 = 5
    IDX_WIDTH = 6
    IDX_HEIGHT = 7

    CROP_AS_TOL = "crop_as_tol"

    def __init__(self,
                 step_theta=1,
                 step_rho=1,
                 threshold=10,
                 angle_error=3,
                 clustering_tol=2,
                 displacement_function=max_displacement
                 ):
        """

        :param step_theta: Theta quantization in [0, 180] range
        :param step_rho: Rho quantization in [0, diag] range
        :param threshold: Hough threshold to be chosen as candidate line
        :param angle_error: Theta error between the mode and other candidate lines
        :param clustering_tol: Rho tolerance to select put a line in the same cluster as the previous one
            You can set 'crop_as_tol' to pick the medium crop size as tol
        :param displacement_function: Function used to choose the square whose center is the centroid of a crop
        """
        self.step_theta = step_theta
        self.step_rho = step_rho
        self.threshold = threshold
        self.crop_detector = load_laweed()
        self.displacement = displacement_function
        self.angle_error = angle_error
        means, stds = WeedMapDatasetInterface.get_mean_std(TRAIN_FOLDERS, CHANNELS, SUBSET)
        self.transform = Normalize(means, stds)
        self.clustering_tol = clustering_tol
        self.mean_crop_size = None
        self.diag_len = None

    def detect_crop(self, input_img):
        input_img = input_img.cuda().unsqueeze(0)
        tensor_img = self.transform(input_img)
        seg = self.crop_detector(tensor_img)
        seg_class = seg.argmax(1)
        seg_class[seg_class == 2] = 255
        seg_class[seg_class == 1] = 255
        return seg_class.squeeze(0).type(torch.uint8)

    def hough_sequential(self, shape, connection_dataframe):
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

    def hough(self, shape, connectivity_tensor: torch.Tensor):
        """
        Modified hough implementation based on regions
        :param shape:  image shape
        :param connectivity_tensor: (N, 8) tensor
        :return: (n_thetas, n_rhos) frequency accumulator
        """
        width, height = shape

        d = np.sqrt(np.square(height) + np.square(width))
        self.diag_len = d + 1

        thetas = torch.arange(0, 180, step=self.step_theta)
        rhos = torch.arange(-d, d + 1, step=self.step_rho)
        cos_thetas = torch.cos(torch.deg2rad(thetas))
        sin_thetas = torch.sin(torch.deg2rad(thetas))

        # Retrieve all the points from dataframe
        points = torch.stack([connectivity_tensor[:, self.IDX_CY] - height / 2,
                              connectivity_tensor[:, self.IDX_CX] - width / 2], dim=1)

        # Calculate the rhos values with dot product
        rho_values = torch.matmul(points.float(), torch.stack([sin_thetas, cos_thetas]))

        # Calculate the displacement for each point to add more lines

        shapes = torch.stack([connectivity_tensor[:, self.IDX_WIDTH],
                              connectivity_tensor[:, self.IDX_HEIGHT]], dim=1)
        displacements = torch.div(shapes.max(dim=1).values, 2).round().int()

        # Hough accumulator array of theta vs rho
        accumulator, edges = histogramdd(
            thetas, rhos, rho_values,
            displacements=displacements
        )
        return accumulator

    def calculate_connectivity_cv2(self, input_img):
        """
        Regions extracted with cv2

        :param input_img: numpy array binary image
        :return: connectivity dataframe
        """
        connectivity = 8
        (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(input_img, connectivity=connectivity,
                                                                                  ltype=cv2.CV_32S)
        num_pixels = stats[:, 4]
        conn = pd.DataFrame({"cx": centroids[:, 0], "cy": centroids[:, 1],
                             "bx": stats[:, 0], "by": stats[:, 1], "width": stats[:, 2], "height": stats[:, 3],
                             "num_pixels": num_pixels}).drop(0)  # Remove first
        self.mean_crop_size = (conn['width'].mean() + conn['height'].mean()) / 2
        return conn

    def calculate_connectivity(self, input_img):
        """
        Regions extracted with PyTorch with GPU
        :param input_img: Binary Tensor located on GPU
        :return: connectivity tensor (N, 6) where each row is (centroid x, centroid y, x0, y0, x1, y1)
        """
        components = connected_components_labeling(input_img)[1:]

        def get_region(label):
            where_t = torch.where(label == components)
            y0 = where_t[0][0]  # First dimension is sorted
            y1 = where_t[0][-1]
            x0 = where_t[-1].min()
            x1 = where_t[-1].max()
            cx = (torch.round((x1 + x0) / 2)).int()
            cy = (torch.round((y1 + y0) / 2)).int()
            return torch.tensor([cx, cy, x0, y0, x1, y1, x1 - x0 + 1, y1 - y0 + 1])
        labels = components.unique()[1:]  # First one is background
        regions = torch.stack(tuple(map(get_region, labels)))
        self.mean_crop_size = ((regions[:, 4] - regions[:, 2]).float().mean()
                               + (regions[:, 5] - regions[:, 3]).float().mean()) / 2
        return regions

    def calculate_mask(self, shape, regions):
        """
        Draws a mask from the region tensor

        :param shape: mask shape
        :param regions: region tensor (N, 6)
        :return: Drew mask
        """
        displ_mask = torch.zeros(shape, dtype=torch.uint8)
        for cx, cy, x0, y0, x1, y1, width, height in regions:
            displacement = self.displacement(width, height)
            displ_mask[
            max(int(cy - displacement), 0): min(int(cy + displacement), shape[0]),
            max(int(cx - displacement), 0): min(int(cx + displacement), shape[1])
            ] = 255
        return displ_mask

    def filter_lines(self, accumulator):
        """
        Filter lines in the accumulator firtsly with a threhold,
        then deleting all the lines with a theta different from the mode with a certain error
        :param accumulator: (n_theta, n_rho) frequency tensor
        :return: sliced accumulator on the rows, theta parallel tensor
        """
        filtered = F.threshold(accumulator, self.threshold, 0)
        mode = filtered.sum(axis=1).argmax().item()
        intervals = get_circular_interval(mode - self.angle_error, mode + self.angle_error + 1, 180 // self.step_theta)
        filtered = torch.concat([filtered[inf:sup] for inf, sup in intervals])
        theta_index = torch.concat([torch.arange(inf, sup) for inf, sup in intervals])
        return filtered, theta_index

    def positivize_rhos(self, accumulator, theta_index):
        """
        Transforms all the lines with a negative rhos with equivalent one with a positive rho.
        It changes the dimension of the accumulator: (n_thetas, n_rhos) -> (2 * n_thetas, n_rhos / 2)
        :param accumulator:
        :param theta_index:
        :return: accumulator and theta index with positive rhos
        """
        chunk_size = int(self.diag_len)
        negatives, positives = accumulator.split(chunk_size, dim=1)
        theta_index = torch.concat([theta_index, theta_index + 180])
        accumulator = torch.concat([positives, negatives.flip(dims=[1])])
        return accumulator, theta_index

    def cluster_lines(self, acc: torch.Tensor, thetas_idcs):
        """
        Cluster lines basing on rhos
        :param acc: frequency accumulator
        :param thetas_idcs: parallel theta tensor
        :return: cluster list indices: index where each cluster starts
        """
        clustering_tol = self.mean_crop_size if self.clustering_tol == self.CROP_AS_TOL else self.clustering_tol

        rhos_thetas = torch.stack(torch.where(acc.T > 0), dim=1)
        thetas_rhos = torch.index_select(rhos_thetas, 1, torch.LongTensor([1, 0]))
        thetas_rhos[:, 0] = thetas_idcs[thetas_rhos[:, 0]]
        cluster_indices = [0]
        if thetas_rhos.shape[0] > 1:
            for i in range(1, thetas_rhos.shape[0]):
                if abs(thetas_rhos[i][1] - thetas_rhos[i - 1][1]) > clustering_tol:
                    cluster_indices.append(i)
        else:
            i = 0
        cluster_indices.append(i+1)
        return thetas_rhos, cluster_indices

    def get_medians(self, theta_rhos: torch.Tensor, cluster_index):
        """
        Get the median lines from each cluster
        :param theta_rhos: tensor for thetas and rhos (N, 2)
        :param cluster_index: list of indices for each cluster start
        :return: medians from each cluster
        """
        if theta_rhos.shape[0] > 0:
            return torch.stack([theta_rhos[(i+j)//2] for i, j in previous_iterator(cluster_index, return_first=False)])
        else:
            return torch.tensor([])

    def predict(self, input_img, return_mean_crop_size=False, return_crop_mask=False):
        """
        Detect rows
        Args:
            input_img: Input tensor
            return_mean_crop_size: tells if return the mean crop size

        Returns:

        """
        width, height = input_img.shape[1:]
        crop_mask = self.detect_crop(input_img)
        connectivity_df = self.calculate_connectivity(crop_mask)
        enhanced_mask = self.calculate_mask((width, height), connectivity_df)
        accumulator = self.hough(enhanced_mask.shape, connectivity_df)
        filtered_acc, theta_index = self.filter_lines(accumulator)
        pos_acc, theta_index = self.positivize_rhos(filtered_acc, theta_index)
        thetas_rhos, clusters_index = self.cluster_lines(pos_acc, theta_index)
        medians = self.get_medians(thetas_rhos, clusters_index)
        res = medians,
        if return_mean_crop_size:
            res += (self.mean_crop_size,)
        if return_crop_mask:
            res += (crop_mask,)
        if len(res) == 1:
            return res[0]
        return res
