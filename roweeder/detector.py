from enum import Enum
from typing import Any
import cv2
import numpy as np
import pandas as pd
import torch
# from ezdl.datasets import WeedMapDatasetInterface
# from ezdl.models.lawin import Laweed, SplitLawin
from torchvision.transforms import Normalize, ToTensor, Compose
from torch.nn import functional as F
from roweeder.histogramdd import histogramdd
from roweeder.utils.utils import (
    get_circular_interval,
    get_medians,
    max_displacement,
    merge_bboxes,
)
from scipy.stats import kstest

LAWEED_PATH = "checkpoints/Laweed-1v1r4nzz_latest.pth"
LAWIN_PATH = "checkpoints/SplitLawin_B1_RedEdge_RGBNIRRE.pth"
TRAIN_FOLDERS = ["000", "001", "002", "004"]
LAWEED_CHANNELS = ["R", "G", "B"]
LAWIN_CHANNELS = ["R", "G", "B", "NIR", "RE"]
SUBSET = "rededge"


def get_vegetation_detector(detector_name, detector_params):
    if detector_name == "NDVIDetector":
        return NDVIVegetationDetector(**detector_params)


class HoughDetectorDict(Enum):
    LINES = "lines"
    CROP_MASK = "crop_mask"
    MEAN_CROP_SIZE = "mean_crop_size"
    COMPONENTS = "components"
    ORIGINAL_LINES = "original_lines"
    REDUCED_THRESHOLD = "reduced_threshold"
    ZERO_REASON = "zero_reason"
    UNIFORM_SIGNIFICANCE = "uniform_significance"


# class LaweedVegetationDetector:
#     def __init__(self):
#         pth = torch.load(LAWEED_PATH)
#         model = Laweed(
#             {
#                 "output_channels": 3,
#                 "num_classes": 3,
#                 "backbone": "MiT-B0",
#                 "backbone_pretrained": True,
#             }
#         )
#         weights = {k[7:]: v for k, v in pth["net"].items()}
#         model.load_state_dict(weights)
#         means, stds = WeedMapDatasetInterface.get_mean_std(
#             TRAIN_FOLDERS, LAWEED_CHANNELS, SUBSET
#         )
#         self.transform = Normalize(means, stds)
#         model.eval()
#         model.cuda()

#         self.model = model

    def __call__(self, img):
        img = self.transform(img)
        seg = self.model(img.unsqueeze(0).cuda())
        seg_class = seg.argmax(1)
        seg_class[seg_class == 2] = 255
        seg_class[seg_class == 1] = 255
        return seg_class


class NDVIVegetationDetector:
    def __init__(self, threshold=0.6, red_idx=0, nir_idx=3) -> None:
        self.threshold = threshold
        self.nir_idx = nir_idx
        self.red_idx = red_idx

    def __call__(self, image=None, ndvi=None) -> Any:
        if ndvi is not None:
            ndvi = ndvi.cuda()
        else:
            image = image.cuda()
            ndvi = (image[self.nir_idx] - image[self.red_idx]) / (image[self.nir_idx] + image[self.red_idx])
        return ((ndvi > self.threshold).type(torch.uint8) * 255).unsqueeze(0)

    def __repr__(self) -> str:
        return f"NDVI Vegetation Detector with threshold {self.threshold}"


# class SplitLawinVegetationDetector:
#     def __init__(self, checkpoint_path=LAWIN_PATH):
#         self.checkpoint_path = checkpoint_path
#         pth = torch.load(checkpoint_path)
#         model = SplitLawin(
#             {
#                 "output_channels": 3,
#                 "num_classes": 3,
#                 "backbone": "MiT-B1",
#                 "side_pretrained": "G",
#                 "backbone_pretrained": True,
#                 "main_channels": 3,
#                 "input_channels": 5,
#             }
#         )
#         weights = {k[7:]: v for k, v in pth["net"].items()}
#         model.load_state_dict(weights)
#         means, stds = WeedMapDatasetInterface.get_mean_std(
#             TRAIN_FOLDERS, LAWIN_CHANNELS, SUBSET
#         )
#         self.transform = Normalize(means, stds)
#         model.eval()
#         model.cuda()

#         self.model = model

#     def __call__(self, img):
#         img = self.transform(img)
#         seg = self.model(img.unsqueeze(0).cuda())
#         seg_class = seg.argmax(1)
#         seg_class[seg_class == 2] = 255
#         seg_class[seg_class == 1] = 255
#         return seg_class

#     def __repr__(self) -> str:
#         return f"SplitLawin Vegetation Detector with checkpoint {self.checkpoint_path}"


class CropRowDetector:
    def __init__(self, crop_detector) -> None:
        self.crop_detector = crop_detector

    def detect_crop(self, input_img):
        seg = self.crop_detector(input_img)
        return seg.squeeze(0).type(torch.uint8)

    def predict(self, input_img):
        raise NotImplementedError

    def predict_from_mask(self, mask):
        raise NotImplementedError


class AbstractHoughCropRowDetector(CropRowDetector):
    CROP_AS_TOL = "crop_as_tol"

    def __init__(
        self,
        step_theta=1,
        step_rho=1,
        threshold=10,
        angle_error=3,
        clustering_tol=2,
        uniform_significance=0.1,
        crop_detector=None,
        theta_reduction_threshold=1.0,
        theta_value=None,
    ):
        super().__init__(crop_detector)
        self.step_theta = step_theta
        self.step_rho = step_rho
        self.threshold = threshold
        self.angle_error = angle_error
        self.clustering_tol = clustering_tol
        self.mean_crop_size = None
        self.diag_len = None
        self.uniform_significance = uniform_significance
        self.theta_reduction_threshold = theta_reduction_threshold
        self.theta_value = theta_value

    def calculate_connectivity(self, input_img):
        """
        Regions extracted with PyTorch with GPU
        :param input_img: Binary Tensor located on GPU
        :return: connectivity tensor (N, 6) where each row is (centroid x, centroid y, x0, y0, x1, y1)
        """
        input_img = input_img.type(torch.uint8)
        if len(input_img.shape) == 3:
            if input_img.shape[0] == 1:
                input_img = input_img.squeeze(0)
            else:
                raise ValueError("Must be 2D tensor")
        components = cv2.connectedComponents(input_img.cpu().numpy())[1:]
        components = torch.tensor(np.array(components)).cuda()

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
        if len(labels) == 0:
            return torch.tensor([]), torch.tensor([])
        regions = torch.stack(tuple(map(get_region, labels)))
        self.mean_crop_size = (
            (regions[:, 4] - regions[:, 2]).float().mean()
            + (regions[:, 5] - regions[:, 3]).float().mean()
        ) / 2
        return components, regions

    def revive_border_lines(
        self, mask, lines, filtered_lines, return_reduced_threshold=False
    ):
        """
        Revive border lines from the filtered lines if they are close to the border
        :param lines: lines tensor
        :param filtered_lines: filtered lines tensor
        :return: revived lines tensor
        """
        raise NotImplementedError("Dismissed")
        # mask = mask.squeeze(0).cpu().type(torch.uint8).numpy()
        # step_theta = self.step_theta * np.pi / 180
        # theta_mode = filtered_lines[:, 1].mode().values.item()
        # print(theta_mode, self.theta_reduction_threshold, step_theta)
        # reduced_threshold = self.theta_reduction_threshold * self.threshold
        # lines = cv2.HoughLines(mask, self.step_rho, step_theta, reduced_threshold)
        # if lines is None:
        #     return torch.tensor([])
        # cartesian_lines = polar_to_cartesian(torch.tensor(lines))
        # rectangle_mask = np.array(
        #     [
        #         (mask.shape[1], 0),
        #         (mask.shape[1], mask.shape[0]),
        #         (0, mask.shape[0]),
        #         (0, 0),
        #     ]
        # )
        # intersection_lens = [
        #     line_in_a_rectangle_len(rectangle_mask, line) for line in cartesian_lines
        # ]
        # intersection_lens = torch.tensor(intersection_lens)
        # min_intersection_len = intersection_lens.argmin()
        # return (lines, reduced_threshold) if return_reduced_threshold else lines

    def test_if_uniform(self, thetas):
        """
        Test if the rho values are uniform
        Args:
            rhos: rho values
        Returns:
            True if uniform
        """
        statistic = kstest(
            (thetas - thetas.min()) / (thetas.max() - thetas.min()), "uniform"
        ).statistic
        return statistic < self.uniform_significance, statistic

    def predict(
        self,
        input_img,
        return_mean_crop_size=False,
        return_crop_mask=False,
        return_components=False,
    ):
        """
        Detect rows
        Args:
            input_img: Input tensor
            return_mean_crop_size: tells if return the mean crop size

        Returns:

        """
        crop_mask = self.detect_crop(input_img)
        return self.predict_from_mask(
            crop_mask, return_mean_crop_size, return_crop_mask, return_components
        )


class ModifiedHoughCropRowDetector(AbstractHoughCropRowDetector):
    IDX_CX = 0
    IDX_CY = 1
    IDX_X0 = 2
    IDX_Y0 = 3
    IDX_X1 = 4
    IDX_Y1 = 5
    IDX_WIDTH = 6
    IDX_HEIGHT = 7

    def __init__(
        self,
        step_theta=1,
        step_rho=1,
        threshold=10,
        angle_error=3,
        clustering_tol=2,
        displacement_function=max_displacement,
        crop_detector=None,
        crop_merge_multiplier=1,
        uniform_significance=0.1,
    ):
        """

        :param step_theta: Theta quantization in [0, 180] range
        :param step_rho: Rho quantization in [0, diag] range
        :param threshold: Hough threshold to be chosen as candidate line
        :param angle_error: Theta error between the mode and other candidate lines
        :param clustering_tol: Rho tolerance to select put a line in the same cluster as the previous one
            You can set 'crop_as_tol' to pick the medium crop size as tol
        :param displacement_function: Function used to choose the square whose center is the centroid of a crop
        :param crop_merge_multiplier: how much multiply from the displacement from the centroid to find other crops for merging into one bbox
        """
        super().__init__(
            crop_detector=crop_detector,
            step_theta=step_theta,
            step_rho=step_rho,
            threshold=threshold,
            angle_error=angle_error,
            clustering_tol=clustering_tol,
            uniform_significance=uniform_significance,
        )
        self.displacement = displacement_function
        self.angle_error = angle_error
        self.clustering_tol = clustering_tol
        self.mean_crop_size = None
        self.diag_len = None
        self.crop_merge_multiplier = crop_merge_multiplier

    def detect_crop(self, input_img):
        seg = self.crop_detector(input_img)
        return seg.squeeze(0).type(torch.uint8)

    def hough_sequential(self, shape, connection_dataframe):
        width, height = shape

        d = np.sqrt(np.square(height) + np.square(width))
        thetas = np.arange(0, 180, step=self.step_theta)
        rhos = np.arange(-d, d, step=self.step_rho)
        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))

        # Hough accumulator array of theta vs rho
        accumulator = np.zeros((len(thetas), len(rhos)))
        for idx, (
            x,
            y,
            bx,
            by,
            pwidth,
            pheight,
            num_pixels,
        ) in connection_dataframe.iterrows():
            displacement = self.displacement(pwidth, pheight)
            point = (y - height / 2, x - width / 2)
            for theta_idx in range(len(thetas)):
                rho = (point[1] * cos_thetas[theta_idx]) + (
                    point[0] * sin_thetas[theta_idx]
                )
                rho_idx = np.argmin(np.abs(rhos - rho))
                accumulator[
                    theta_idx, rho_idx - displacement : rho_idx + displacement + 1
                ] += 1
        theta_vec, rho_vec = np.where(accumulator > self.threshold)
        return pd.DataFrame(
            {"theta": theta_vec * self.step_theta, "rho": rho_vec * self.step_rho - d}
        )

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
        points = torch.stack(
            [
                connectivity_tensor[:, self.IDX_CY] - height / 2,
                connectivity_tensor[:, self.IDX_CX] - width / 2,
            ],
            dim=1,
        )

        # Calculate the rhos values with dot product
        rho_values = torch.matmul(points.float(), torch.stack([sin_thetas, cos_thetas]))

        # Calculate the displacement for each point to add more lines

        shapes = torch.stack(
            [
                connectivity_tensor[:, self.IDX_WIDTH],
                connectivity_tensor[:, self.IDX_HEIGHT],
            ],
            dim=1,
        )
        displacements = torch.div(shapes.max(dim=1).values, 2).round().int()

        # Hough accumulator array of theta vs rho
        accumulator, edges = histogramdd(
            thetas, rhos, rho_values, displacements=displacements
        )
        return accumulator

    def calculate_connectivity_cv2(self, input_img):
        """
        Regions extracted with cv2

        :param input_img: numpy array binary image
        :return: connectivity dataframe
        """
        connectivity = 8
        (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
            input_img, connectivity=connectivity, ltype=cv2.CV_32S
        )
        num_pixels = stats[:, 4]
        conn = pd.DataFrame(
            {
                "cx": centroids[:, 0],
                "cy": centroids[:, 1],
                "bx": stats[:, 0],
                "by": stats[:, 1],
                "width": stats[:, 2],
                "height": stats[:, 3],
                "num_pixels": num_pixels,
            }
        ).drop(
            0
        )  # Remove first
        self.mean_crop_size = (conn["width"].mean() + conn["height"].mean()) / 2
        return conn

    def calculate_mask(self, shape, regions):
        """
        Draws a mask from the region tensor

        :param shape: mask shape
        :param regions: region tensor (N, 8)
        :return: Drew mask
        """
        displ_mask = torch.zeros(shape, dtype=torch.uint8)
        for cx, cy, x0, y0, x1, y1, width, height in regions:
            displacement = self.displacement(width, height)
            displ_mask[
                max(int(cy - displacement), 0) : min(int(cy + displacement), shape[0]),
                max(int(cx - displacement), 0) : min(int(cx + displacement), shape[1]),
            ] = 255
        return displ_mask

    def increase_recall(self, regions):
        """
        Temporary Failed
        """

        def get_neighbours(cx, cy, regions, threshold):
            return ((regions[:, self.IDX_CX] - cx).abs() < threshold) & (
                (regions[:, self.IDX_CY] - cy).abs() < threshold
            )

        presence = torch.ones(regions.shape[0], dtype=torch.bool)
        for i in range(regions.shape[0]):
            cx, cy, x0, y0, x1, y1, width, height = regions[i]
            neighbours = get_neighbours(
                cx,
                cy,
                regions[presence],
                self.displacement(width, height) * self.crop_merge_multiplier,
            )
            if neighbours.sum() > 1:  # Itself
                neighbours = presence.argwhere().squeeze(1)[
                    neighbours
                ]  # Get back indices to original tensor
                new_bbox = merge_bboxes(regions[neighbours])
                regions[neighbours[-1]] = new_bbox
                presence[neighbours[:-1]] = False
        return regions[presence].contiguous()

    def filter_lines(self, accumulator):
        """
        Filter lines in the accumulator firstly with a threshold,
        then deleting all the lines with a theta different from the mode with a certain error
        :param accumulator: (n_theta, n_rho) frequency tensor
        :return: sliced accumulator on the rows, theta parallel tensor
        """
        filtered = F.threshold(accumulator, self.threshold, 0)
        mode = filtered.sum(axis=1).argmax().item()
        intervals = get_circular_interval(
            mode - self.angle_error, mode + self.angle_error + 1, 180 // self.step_theta
        )
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
        clustering_tol = (
            self.mean_crop_size
            if self.clustering_tol == self.CROP_AS_TOL
            else self.clustering_tol
        )

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
        cluster_indices.append(i + 1)
        return thetas_rhos, cluster_indices

    def predict_from_mask(
        self,
        mask,
        return_mean_crop_size=False,
        return_crop_mask=False,
        return_components=False,
    ):
        """
        Detect rows
        Args:
            input_img: Input tensor
            return_mean_crop_size: tells if return the mean crop size

        Returns:

        """
        if len(mask.shape) == 3:
            if mask.shape[0] == 1:
                mask = mask.squeeze(0)
            else:
                raise ValueError("Must be 2D tensor")
        width, height = mask.shape
        crop_mask = mask.cuda()
        components, connectivity_df = self.calculate_connectivity(crop_mask)
        enhanced_mask = self.calculate_mask((width, height), connectivity_df)
        accumulator = self.hough(enhanced_mask.shape, connectivity_df)
        filtered_acc, theta_index = self.filter_lines(accumulator)
        pos_acc, theta_index = self.positivize_rhos(filtered_acc, theta_index)
        thetas_rhos, clusters_index = self.cluster_lines(pos_acc, theta_index)
        medians = get_medians(thetas_rhos, clusters_index)
        res = (medians,)
        if return_mean_crop_size:
            res += (self.mean_crop_size,)
        if return_crop_mask:
            res += (crop_mask,)
        if return_components:
            res += (components,)
        return res[0] if len(res) == 1 else res


class HoughCropRowDetector(AbstractHoughCropRowDetector):
    def hough(self, mask):
        """
        Apply hough transform to the mask
        :param mask: mask tensor
        :return: (rho, theta) tensor
        """
        step_theta = self.step_theta * np.pi / 180
        mask = mask.squeeze(0).cpu().type(torch.uint8).numpy()
        lines = cv2.HoughLines(mask, self.step_rho, step_theta, self.threshold)
        return torch.tensor([]) if lines is None else torch.tensor(lines).squeeze(1)

    def filter_lines(self, lines):
        """
        Filter lines that doesn't have the theta mode
        :param lines: (rho, theta) tensor
        :return: filtered lines
        """
        thetas = lines[:, 1]
        n_bins = int(180 / self.step_theta)
        hist = torch.histogram(thetas, bins=n_bins, range=(0, np.pi))
        theta_mode = (
            hist.bin_edges[hist.hist.argmax()]
            if self.theta_value is None
            else self.theta_value
        )
        step_theta = self.step_theta * np.pi / 180
        thetas_in_bin = (thetas >= theta_mode - step_theta) & (
            thetas <= theta_mode + step_theta
        )
        return lines[thetas_in_bin]

    def cluster_lines(self, lines: torch.Tensor):
        """
        Cluster lines basing on rhos with a certain tolerance
        :param lines: (rho, theta) tensor
        :return: cluster list indices: index where each cluster starts
        """
        clustering_tol = (
            self.mean_crop_size
            if self.clustering_tol == self.CROP_AS_TOL
            else self.clustering_tol
        )
        rhos = lines[:, 0]
        sorted_lines = lines[torch.argsort(rhos)]
        cluster_indices = [0]
        if sorted_lines.shape[0] > 1:
            for i in range(1, sorted_lines.shape[0]):
                if abs(sorted_lines[i][0] - sorted_lines[i - 1][0]) > clustering_tol:
                    cluster_indices.append(i)
        else:
            i = 0
        cluster_indices.append(i + 1)
        return sorted_lines, cluster_indices

    def predict_from_mask(
        self,
        mask,
    ):
        """
        Detect rows
        Args:
            input_img: Input tensor
            return_mean_crop_size: tells if return the mean crop size

        Returns:

        """
        crop_mask = mask.cuda()
        zero_reason = None
        uniform_statistic = None
        original_lines = torch.tensor([])
        components, regions = self.calculate_connectivity(
            crop_mask
        )  # To calculate the mean crop size
        reduced_threshold = None
        if len(regions) == 0:
            zero_reason = "No components"
            res = torch.tensor([])
        else:
            original_lines = self.hough(crop_mask)
            if len(original_lines) == 0:
                zero_reason = "No lines thresholded"
                res = torch.tensor([])
            else:
                is_uniform, uniform_statistic = self.test_if_uniform(
                    original_lines[:, 1]
                )
                if is_uniform:
                    zero_reason = "Uniform"
                    res = torch.tensor([])

                filtered_lines = self.filter_lines(original_lines)
                thetas_rhos, clusters_index = self.cluster_lines(filtered_lines)
                medians = get_medians(thetas_rhos, clusters_index)
                res = medians

        return_dict = {
            HoughDetectorDict.LINES: res,
            HoughDetectorDict.MEAN_CROP_SIZE: self.mean_crop_size,
            HoughDetectorDict.CROP_MASK: crop_mask,
            HoughDetectorDict.COMPONENTS: components,
            HoughDetectorDict.ORIGINAL_LINES: original_lines,
            HoughDetectorDict.REDUCED_THRESHOLD: reduced_threshold,
            HoughDetectorDict.ZERO_REASON: zero_reason,
            HoughDetectorDict.UNIFORM_SIGNIFICANCE: uniform_statistic,
        }

        return return_dict
