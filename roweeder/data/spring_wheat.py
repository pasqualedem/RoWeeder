import os
from typing import Any, Union, Iterable, Mapping

import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from logging import getLogger

from roweeder.data.stats import STATS
from sklearn.model_selection import train_test_split

from roweeder.data import stats
from roweeder.utils.utils import remove_suffix

logger = getLogger(__name__)


class SpringWheatDatasetInterface:
    STATS = stats.STATS

    def __init__(self, dataset_params):
        mean, std = self.get_mean_std()

        self.lib_dataset_params = {
            'mean': mean,
            'std': std,
        }

        input_transform = [
            transforms.Normalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std']),
        ]

        test_transform = [
            transforms.Normalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std']),
        ]

        if core_utils.get_param(self.dataset_params, 'size', default_val='same') != 'same':
            resize = transforms.Resize(size=core_utils.get_param(self.dataset_params, 'size', default_val='same'))
            input_transform.append(resize)
            test_transform.append(resize)

        input_transform = transforms.Compose(input_transform)

        self.trainset = SpringWheatDataset(root=self.dataset_params.root,
                                           transform=input_transform,
                                           return_path=dataset_params['return_path'])

    def undo_preprocess(self, x):
        return (Denormalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std'])(x) * 255).type(torch.uint8)

    @classmethod
    def get_mean_std(cls):
        return zip([(d['mean'], d['std']) for d in cls.STATS.values()])


class SpringWheatDataset(VisionDataset):
    CLASS_LABELS = {0: "background", 1: "crop", 2: 'weed'}
    classes = ['background', 'crop', 'weed']

    IMG_EXTENSION = '.JPG'

    def __init__(self,
                 root: str,
                 transform: callable,
                 return_path: bool = False,
                 ):
        """
        Initialize a SpringWheatDataset object.

        :param root: The root directory.
        :param transform: The transform to apply to the data.
        :param return_path: Whether to return the path.
        """
        super().__init__(root=root)

        self.path = root
        self.transform = transform if transform is not None else lambda x: x
        self.return_name = return_path
        self.files = os.listdir(self.path)

    def __getitem__(self, i) -> Any:
        img_path = os.path.join(self.path, self.files[i])
        img = Image.open(img_path)
        if self.return_name:
            return self.transform(F.to_tensor(img)), img_path
        return self.transform(F.to_tensor(img))

    def __len__(self) -> int:
        """
        Get the number of batches.

        :return: The number of batches.
        """
        return len(self.files)


class SpringWheatMaskedDataset(SpringWheatDataset):
    MASK_SUFFIX = '_cropmask.png'
    CSV_SUFFIX = '_mask.csv'

    def __init__(self,
                 root: str,
                 transform: callable,
                 return_path: bool = False,
                 return_img: bool = True
                 ):
        super().__init__(root, transform, return_path)
        self.files = list(
            {
                remove_suffix(img, self.MASK_SUFFIX)
                for img in self.files
                if img.endswith(self.MASK_SUFFIX)
            }
        )
        self.return_img = return_img

    def __getitem__(self, i) -> Any:
        img_path = os.path.join(self.path, self.files[i] + self.IMG_EXTENSION)
        mask_path = img_path.replace(self.IMG_EXTENSION, self.MASK_SUFFIX)
        mask = F.pil_to_tensor(Image.open(mask_path))
        if self.return_img:
            img = self.transform(F.to_tensor(Image.open(img_path)))
            return (img, mask, img_path) if self.return_name else (img, mask)
        return (mask, img_path) if self.return_name else mask
