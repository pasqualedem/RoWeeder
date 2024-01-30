from itertools import tee, chain
from urllib.parse import urlunparse, urlparse
import numpy as np

import torch


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

    # Combine m and b into a new matrix
    cartesian_matrix = torch.stack((m, b), dim=1)

    return cartesian_matrix


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
