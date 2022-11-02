import itertools
import os
from typing import Any, Union, Iterable, Mapping

import functools

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from ezdl.transforms import \
    PairRandomCrop, SegOneHot, ToLong, FixValue, Denormalize, PairRandomFlip, squeeze0, \
    PairFlip, PairFourCrop
from data.stats import STATS
from sklearn.model_selection import train_test_split

from torch.utils.data.distributed import DistributedSampler
from super_gradients.training import utils as core_utils
from super_gradients.training.datasets.mixup import CollateMixup
from super_gradients.training.exceptions.dataset_exceptions import IllegalDatasetParameterException
from super_gradients.common.abstractions.abstract_logger import get_logger
from ezdl.data import DatasetInterface

from data import stats

logger = get_logger(__name__)


class WeedMapDatasetInterface(DatasetInterface):
    STATS = stats.STATS

    def __init__(self, dataset_params):
        super(WeedMapDatasetInterface, self).__init__(dataset_params)
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
        return cls.STATS

    def build_data_loaders(self, batch_size_factor=1, num_workers=8, train_batch_size=None, val_batch_size=None,
                           test_batch_size=None, distributed_sampler: bool = False):
        """

        define train, val (and optionally test) loaders. The method deals separately with distributed training and standard
        (non distributed, or parallel training). In the case of distributed training we need to rely on distributed
        samplers.
        :param batch_size_factor: int - factor to multiply the batch size (usually for multi gpu)
        :param num_workers: int - number of workers (parallel processes) for dataloaders
        :param train_batch_size: int - batch size for train loader, if None will be taken from dataset_params
        :param val_batch_size: int - batch size for val loader, if None will be taken from dataset_params
        :param distributed_sampler: boolean flag for distributed training mode
        :return: train_loader, val_loader, classes: list of classes
        """
        # CHANGE THE BATCH SIZE ACCORDING TO THE NUMBER OF DEVICES - ONLY IN NON-DISTRIBUED TRAINING MODE
        # IN DISTRIBUTED MODE WE NEED DISTRIBUTED SAMPLERS
        # NO SHUFFLE IN DISTRIBUTED TRAINING
        if distributed_sampler:
            self.batch_size_factor = 1
            train_sampler = DistributedSampler(self.trainset)
            val_sampler = DistributedSampler(self.valset)
            test_sampler = DistributedSampler(self.testset) if self.testset is not None else None
            train_shuffle = False
        else:
            self.batch_size_factor = batch_size_factor
            train_sampler = None
            val_sampler = None
            test_sampler = None
            train_shuffle = True

        if train_batch_size is None:
            train_batch_size = self.dataset_params.batch_size * self.batch_size_factor
        if val_batch_size is None:
            val_batch_size = self.dataset_params.val_batch_size * self.batch_size_factor
        if test_batch_size is None:
            test_batch_size = self.dataset_params.test_batch_size * self.batch_size_factor

        train_loader_drop_last = core_utils.get_param(self.dataset_params, 'train_loader_drop_last', default_val=False)

        cutmix = core_utils.get_param(self.dataset_params, 'cutmix', False)
        cutmix_params = core_utils.get_param(self.dataset_params, 'cutmix_params')

        # WRAPPING collate_fn
        train_collate_fn = core_utils.get_param(self.trainset, 'collate_fn')
        val_collate_fn = core_utils.get_param(self.valset, 'collate_fn')
        test_collate_fn = core_utils.get_param(self.testset, 'collate_fn')

        if cutmix and train_collate_fn is not None:
            raise IllegalDatasetParameterException("cutmix and collate function cannot be used together")

        if cutmix:
            # FIXME - cutmix should be available only in classification dataset. once we make sure all classification
            # datasets inherit from the same super class, we should move cutmix code to that class
            logger.warning("Cutmix/mixup was enabled. This feature is currently supported only "
                           "for classification datasets.")
            train_collate_fn = CollateMixup(**cutmix_params)

        # FIXME - UNDERSTAND IF THE num_replicas VARIBALE IS NEEDED
        # train_sampler = DistributedSampler(self.trainset,
        #                                    num_replicas=distributed_gpus_num) if distributed_sampler else None
        # val_sampler = DistributedSampler(self.valset,
        #                                   num_replicas=distributed_gpus_num) if distributed_sampler else None
        pw = num_workers > 0
        self.train_loader = torch.utils.data.DataLoader(self.trainset,
                                                        batch_size=train_batch_size,
                                                        shuffle=train_shuffle,
                                                        num_workers=num_workers,
                                                        pin_memory=True,
                                                        sampler=train_sampler,
                                                        collate_fn=train_collate_fn,
                                                        drop_last=train_loader_drop_last,
                                                        persistent_workers=pw)

        self.val_loader = torch.utils.data.DataLoader(self.valset,
                                                      batch_size=val_batch_size,
                                                      shuffle=False,
                                                      num_workers=num_workers,
                                                      pin_memory=True,
                                                      sampler=val_sampler,
                                                      collate_fn=val_collate_fn,
                                                      persistent_workers=pw)

        if self.testset is not None:
            self.test_loader = torch.utils.data.DataLoader(self.testset,
                                                           batch_size=test_batch_size,
                                                           shuffle=False,
                                                           num_workers=num_workers,
                                                           pin_memory=True,
                                                           sampler=test_sampler,
                                                           collate_fn=test_collate_fn,
                                                           persistent_workers=pw)

        self.classes = self.trainset.classes


class SpringWheatDataset(VisionDataset):
    CLASS_LABELS = {0: "background", 1: "crop", 2: 'weed'}
    classes = ['background', 'crop', 'weed']

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
        self.transform = transform
        self.return_name = return_path
        self.files = os.listdir(self.path)
        self.len = len(self.files)

    def __getitem__(self, i) -> Any:
        img = Image.open(
            os.path.join(self.path, self.files[i])
        )
        return self.transform(F.to_tensor(img))

    def __len__(self) -> int:
        """
        Get the number of batches.

        :return: The number of batches.
        """
        return self.len
