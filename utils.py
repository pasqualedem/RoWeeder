from itertools import tee, chain
from urllib.parse import urlunparse, urlparse

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
    x1 = bboxes[:, 0].min()
    y1 = bboxes[:, 1].min()
    x2 = bboxes[:, 2].max()
    y2 = bboxes[:, 3].max()
    return torch.tensor([x1, y1, x2, y2])
