from enum import Enum
import os
import cv2
import time
import yaml
import torch
import numpy as np
import importlib
import collections.abc
from datetime import datetime
from inspect import signature
from ruamel.yaml import YAML, comments
from io import StringIO
from typing import Mapping
from itertools import tee, chain
from urllib.parse import urlunparse, urlparse


FLOAT_PRECISIONS = {
    "fp32": torch.float32,
    "fp64": torch.float64,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class StrEnum(str, Enum):
    pass


def load_yaml(file_path):
    with open(file_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file.read())
        

def write_yaml(data: dict, file_path: str = None, file=None):
    """ Write a dictionary to a YAML file.

    Args:
        data (dict): the data to write
        file_path (str): the path to the file
        file: the file object to write to (esclusive with file_path)
    """
    if file is not None:
        file.write(yaml.dump(data))
        return
    if file_path is None:
        raise ValueError("file_path or file must be specified")
    try:
        with open(file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


def get_module_class_from_path(path):
    path = os.path.normpath(path)
    splitted = path.split(os.sep)
    module = ".".join(splitted[:-1])
    cls = splitted[-1]
    return module, cls


def update_collection(collec, value, key=None):
    if isinstance(collec, dict):
        if isinstance(value, dict):
            for keyv, valuev in value.items():
                collec = update_collection(collec, valuev, keyv)
        elif key is not None:
            if value is not None:
                collec[key] = value
        else:
            collec = {**collec, **value} if value is not None else collec
    else:
        collec = value if value is not None else collec
    return collec


def nested_dict_update(d, u):
    if u is not None:
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = nested_dict_update(d.get(k) or {}, v)
            else:
                d[k] = v
    return d


def instantiate_class(name, params):
    module, cls = get_module_class_from_path(name)
    imp_module = importlib.import_module(module)
    imp_cls = getattr(imp_module, cls)
    if (
        len(signature(imp_cls).parameters.keys()) == 1
        and "params" in list(signature(imp_cls).parameters.keys())[0]
    ):
        return imp_cls(params)
    return imp_cls(**params)


def substitute_values(x: torch.Tensor, values, unique=None):
    """
    Substitute values in a tensor with the given values
    :param x: the tensor
    :param unique: the unique values to substitute
    :param values: the values to substitute with
    :return: the tensor with the values substituted
    """
    if unique is None:
        unique = x.unique()
    lt = torch.full((unique.max() + 1,), -1, dtype=values.dtype, device=x.device)
    lt[unique] = values
    return lt[x]


# def load_yaml(path, return_string=False):
#     if hasattr(path, "readlines"):
#         d = convert_commentedmap_to_dict(YAML().load(path))
#         if return_string:
#             path.seek(0)
#             return d, path.read().decode("utf-8")
#     with open(path, "r") as param_stream:
#         d = convert_commentedmap_to_dict(YAML().load(param_stream))
#         if return_string:
#             param_stream.seek(0)
#             return d, str(param_stream.read())
#     return d


def convert_commentedmap_to_dict(data):
    """
    Recursive function to convert CommentedMap to dict
    """
    if isinstance(data, comments.CommentedMap):
        result = {}
        for key, value in data.items():
            result[key] = convert_commentedmap_to_dict(value)
        return result
    elif isinstance(data, list):
        return [convert_commentedmap_to_dict(item) for item in data]
    else:
        return data


def log_every_n(image_idx: int, n: int):
    if n is None:
        return False
    return image_idx % n == 0


def dict_to_yaml_string(mapping: Mapping) -> str:
    """
    Convert a nested dictionary or list to a string
    """
    string_stream = StringIO()
    yaml = YAML()
    yaml.dump(mapping, string_stream)
    output_str = string_stream.getvalue()
    string_stream.close()
    return output_str


def get_checkpoints_dir_path(
    project_name: str, group_name: str, ckpt_root_dir: str = None
):
    """Creating the checkpoint directory of a given experiment.
    :param experiment_name:     Name of the experiment.
    :param ckpt_root_dir:       Local root directory path where all experiment logging directories will
                                reside. When none is give, it is assumed that pkg_resources.resource_filename('checkpoints', "")
                                exists and will be used.
    :return:                    checkpoints_dir_path
    """
    if ckpt_root_dir:
        return os.path.join(ckpt_root_dir, project_name, group_name)


def get_timestamp():
    # Get the current timestamp
    timestamp = time.time()  # replace this with your timestamp or use time.time() for current time

    # Convert timestamp to a datetime object
    dt_object = datetime.fromtimestamp(timestamp)

    # Format the datetime object as a folder-friendly string
    return dt_object.strftime("%Y%m%d_%H%M%S")

def find_divisor_pairs(number):
    divisor_pairs = []
    
    for i in range(1, int(number**0.5) + 1):
        if number % i == 0:
            divisor_pairs.append((i, number // i))
    
    return divisor_pairs


def get_divisors(n):
    """
    Returns a list of divisors of a given number.

    Args:
        n (int): The number to find divisors for.

    Returns:
        list: A list of divisors of the given number.
    """
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


class RunningAverage:
    def __init__(self):
        self.accumulator = 0
        self.steps = 0
        
    def update(self, value):
        self.accumulator += value
        self.steps += 1
        
    def compute(self):
        return self.accumulator / self.steps


def previous_iterator(some_iterable, return_first=True):
    prevs, items = tee(some_iterable, 2)
    prevs = chain([None], prevs)
    it = zip(prevs, items)
    if not return_first:
        next(it)
    return it


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def change_url_host(source, destination):
    _, host, _, _, _, _ = urlparse(source)
    scheme, _, path, params, query, fragment = urlparse(destination)
    return urlunparse((scheme, host, path, params, query, fragment))



def tensor_intersection(x, y):
    combined = torch.cat([x, y])
    counts = combined.unique(return_counts=True)
    return counts[0][counts[1][counts[1] > 1]]


def remove_element_from_tensor(x, index):
    return torch.cat([x[:index], x[index+1:]])


def merge_bboxes(bboxes):
    """
    Merge bboxes into a single bbox.
    """
    x1 = bboxes[:, 2].min()
    y1 = bboxes[:, 3].min()
    x2 = bboxes[:, 4].max()
    y2 = bboxes[:, 5].max()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return torch.tensor([cx, cy, x1, y1, x2, y2, width, height])


def intersection_point(l1, l2):
    m1, b1 = l1
    m2, b2 = l2
    # Check if one of the lines is vertical
    if m1 is None:
        x = b1
        y = m2 * x + b2
    elif m2 is None:
        x = b2
        y = m1 * x + b1
    else:
        # Calculate the x-coordinate of the intersection point
        x = (b2 - b1) / (m1 - m2)

        # Use one of the line equations to find the y-coordinate
        y = m1 * x + b1

    return x, y


def polar_to_cartesian(theta_rho_matrix):
    # Extract theta and rho from the input matrix
    theta = theta_rho_matrix[:, 0]
    rho = theta_rho_matrix[:, 1]

    # Convert polar coordinates to Cartesian coordinates
    m = torch.tan(theta)
    b = rho / torch.cos(theta)

    return torch.stack((m, b), dim=1)


def rectangle_line_intersection(rectangle, line):
    (p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y) = rectangle

    up_line = (0, p1y)
    right_line = (None, p2x)
    bottom_line = (0, p3y)
    left_line = (None, p4x)

    up_intersect = intersection_point(up_line, line)
    right_intersect = intersection_point(right_line, line)
    bottom_intersect = intersection_point(bottom_line, line)
    left_intersect = intersection_point(left_line, line)

    on_up = (
        up_intersect is not None and up_intersect[0] > p1x and up_intersect[0] < p2x
    )
    on_right = (
        right_intersect is not None
        and right_intersect[1] >= p3y
        and right_intersect[1] <= p2y
    )
    on_bottom = (
        bottom_intersect is not None
        and bottom_intersect[0] > p4x
        and bottom_intersect[0] < p3x
    )
    on_left = (
        left_intersect is not None
        and left_intersect[1] >= p4y
        and left_intersect[1] <= p1y
    )

    if on_up + on_right + on_bottom + on_left > 2:
        raise ValueError("Line intersects more than 2 sides of the rectangle")
    elif on_up + on_right + on_bottom + on_left < 2:
        return None, None

    p_int_1, p_int_2 = np.array(
        [up_intersect, right_intersect, bottom_intersect, left_intersect]
    )[np.array([on_up, on_right, on_bottom, on_left])]
    return p_int_1, p_int_2


def line_in_a_rectangle_len(rectangle, line):
    p1_int, p2_int = rectangle_line_intersection(rectangle, line)
    return 0 if p1_int is None else np.linalg.norm(p1_int - p2_int)


def line_intersection(segment_start, segment_end, line_slope, line_intercept):
    x1, y1 = segment_start
    x2, y2 = segment_end

    # Check if the segment is vertical
    if x1 == x2:
        # Check if the line is also vertical (no intersection)
        if line_slope is None:
            return None
        # Calculate the x-coordinate of the intersection
        x_intersect = x1
        # Calculate the y-coordinate of the intersection using the line equation
        y_intersect = line_slope * x_intersect + line_intercept
    else:
        # Calculate the slope of the segment
        segment_slope = (y2 - y1) / (x2 - x1)

        # Check if the segment and line are parallel (no intersection)
        if segment_slope == line_slope:
            return None

        # Calculate the x-coordinate of the intersection
        x_intersect = (line_intercept - y1 + segment_slope * x1) / (
            segment_slope - line_slope
        )

        # Calculate the y-coordinate of the intersection using the segment equation
        y_intersect = segment_slope * x_intersect + y1

    # Check if the intersection point is within the segment bounds
    if (x1 <= x_intersect <= x2 or x2 <= x_intersect <= x1) and (
        y1 <= y_intersect <= y2 or y2 <= y_intersect <= y1
    ):
        return (x_intersect, y_intersect)
    else:
        return None


def get_circular_interval(inf, sup, interval_max):
    if sup < inf:
        return (0, sup), (inf, interval_max)
    if sup > interval_max:
        return (0, sup - interval_max), (inf, interval_max)
    if inf < 0:
        return (0, sup), (interval_max + inf, interval_max)
    return ((inf, sup),)


def max_displacement(width, height):
    return int(width / 2 if width > height else height / 2)


def mean_displacement(width, height):
    return int((width + height) / 4)


def get_medians(theta_rhos: torch.Tensor, cluster_index):
    """
    Get the median lines from each cluster
    :param theta_rhos: tensor for thetas and rhos (N, 2)
    :param cluster_index: list of indices for each cluster start
    :return: medians from each cluster
    """
    if theta_rhos.shape[0] > 0:
        return torch.stack(
            [
                theta_rhos[(i + j) // 2]
                for i, j in previous_iterator(cluster_index, return_first=False)
            ]
        )
    else:
        return torch.tensor([])


def get_line_boxes(theta, rho, is_deg, img_width, img_height):
    if is_deg:
        theta = np.deg2rad(theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = (a * rho) + img_width / 2
    y0 = (b * rho) + img_height / 2
    x1 = int(x0 + img_width * (-b))
    y1 = int(y0 + img_height * (a))
    x2 = int(x0 - img_width * (-b))
    y2 = int(y0 - img_height * (a))
    return (x1, y1), (x2, y2)


def get_square_from_lines(img_array, theta, rho, displacement, width, height):
    p1, p2 = get_line_boxes(theta, rho + displacement, True, width, height)
    p3, p4 = get_line_boxes(theta, rho - displacement, True, width, height)
    rect = cv2.minAreaRect(np.array([p1, p2, p3, p4]))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img_array, [box], 0, 255, -1)
    return img_array


class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        if isinstance(d, tuple):
            for t in d:
                setattr(self, t[0], t[1])
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)