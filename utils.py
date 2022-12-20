from itertools import tee, chain


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