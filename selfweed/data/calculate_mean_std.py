import multiprocessing

import torch
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import transforms

from data.spring_wheat import SpringWheatDataset

WORKERS = multiprocessing.cpu_count()
BATCH_SIZE = 16
WIDTH = 1024
HEIGHT = 1024


def calculate(root):
    """
    Calculate the mean and the standard deviation of a dataset
    """

    channels = ['R', 'G', 'B']
    sq = SpringWheatDataset(root,
                            transform=lambda x: x,
                            )
    count = len(sq) * WIDTH * HEIGHT

    # data loader
    image_loader = DataLoader(sq,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)

    # placeholders
    psum = torch.zeros(len(channels))
    psum_sq = torch.zeros(len(channels))

    # loop through images
    for input in tqdm(image_loader):
        psum += input.sum(axis=[0, 2, 3])
        psum_sq += (input ** 2).sum(axis=[0, 2, 3])

    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    d = {}
    for i in range(len(channels)):
        d[channels[i]] = {
            'mean': total_mean[i].item(),
            'std': total_std[i].item(),
            'sum': psum[i].item(),
            'sum_sq': psum_sq[i].item(),
        }
    d['count'] = count
    return d


if __name__ == '__main__':
    root = "./dataset"
    d = calculate(root)
    print(json.dumps(d, indent=4))
