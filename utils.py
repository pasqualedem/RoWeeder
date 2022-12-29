from itertools import tee, chain
from urllib.parse import urlunparse, urlparse


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
