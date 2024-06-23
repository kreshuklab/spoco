import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from spoco.datasets.utils import FilterSliceBuilder
from spoco.transforms import Standardize, Relabel, GaussianBlurNp

LABEL_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        Relabel(run_cc=False),
        transforms.ToTensor()
    ]
)

TEST_LABEL_TRANSFORM = transforms.Compose(
    [
        Relabel(run_cc=False),
        transforms.ToTensor()
    ]
)

EXTENDED_TRANSFORM = transforms.Compose(
    [
        GaussianBlurNp(execution_probability=1.0),
        transforms.ToTensor()
    ]
)


def _build_slices(patch_shape, stride_shape, raw_dataset, label_dataset):
    slice_builder = FilterSliceBuilder(raw_dataset, label_dataset, patch_shape, stride_shape)
    return slice_builder.raw_slices, slice_builder.label_slices


class VolumetricH5Dataset(Dataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.

    Args:
        file_path (str): path to H5 file containing raw data and label data
        phase (str): 'train' for training, 'val' for validation, 'test' for testing
        patch_shape (tuple[int, int, int]): shape of the patches to be extracted
        stride_shape (tuple[int, int, int]): shape of the stride when extracting patches
        raw_internal_path (str): H5 internal path to the raw dataset, default is 'raw'
        label_internal_path (str): H5 internal path to the label dataset, default is 'label'
        global_normalization (bool): if True, the mean and std of the raw data will be calculated over the whole dataset
        instances_ratio (float): ratio of instances to be used for self-training
        spoco (bool): if True, the dataset is used for spoco training
    """

    def __init__(self, file_path, phase, patch_shape, stride_shape,
                 raw_internal_path='raw', label_internal_path='label', global_normalization=True,
                 instance_ratio=None, spoco=False):
        assert phase in ['train', 'val', 'test']

        self.phase = phase
        self.file_path = file_path
        self.raw_internal_path = raw_internal_path
        self.label_internal_path = label_internal_path
        self.instance_ratio = instance_ratio
        self.spoco = spoco

        if phase == 'test':
            with h5py.File(file_path, 'r') as f:
                self.raw = f[raw_internal_path][:]
                self.label = None
        else:
            with (h5py.File(file_path, 'r') as f):
                self.raw = f[raw_internal_path][:]
                if phase == 'train' and instance_ratio is not None:
                    label_internal_path = f'label_{instance_ratio}'
                    assert label_internal_path in f, (f"Label dataset {label_internal_path} "
                                                      f"with instance ratio {instance_ratio} not found")
                self.label = f[label_internal_path][:]
                # make sure the raw and label datasets have the same shape
                assert self.raw.shape[-3:] == self.label.shape[-3:], "Raw and label datasets must have the same shape"

        self.raw_slices, self.label_slices = _build_slices(patch_shape, stride_shape, self.raw, self.label)

        if global_normalization:
            with h5py.File(file_path, 'r') as f:
                raw = f[raw_internal_path][:]
                raw_mean, raw_std = raw.mean(), raw.std()
        else:
            raw_mean, raw_std = None, None

        self.base_raw_transform = transforms.Compose(
            [
                Standardize(mean=raw_mean, std=raw_std),
                transforms.RandomHorizontalFlip()
            ]
        )
        # raw transform for validation and testing
        self.raw_transform = transforms.Compose(
            [
                Standardize(mean=raw_mean, std=raw_std),
                transforms.ToTensor()
            ]
        )

        self.patch_count = len(self.raw_slices)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]

        if self.phase == 'train':
            seed = np.random.randint(np.iinfo('int32').max)
            random.seed(seed)
            torch.manual_seed(seed)
            raw_patch_transformed = self.base_raw_transform(self.raw[raw_idx])
            random.seed(seed)
            torch.manual_seed(seed)
            label_idx = self.label_slices[idx]
            label_patch_transformed = LABEL_TRANSFORM(self.label[label_idx])[0]
            if self.spoco:
                raw_patch_transformed1 = EXTENDED_TRANSFORM(raw_patch_transformed)
                return raw_patch_transformed, raw_patch_transformed1, label_patch_transformed
            else:
                return raw_patch_transformed, label_patch_transformed
        elif self.phase == 'val':
            raw_patch_transformed = self.raw_transform(self.raw[raw_idx])
            label_idx = self.label_slices[idx]
            label_patch_transformed = TEST_LABEL_TRANSFORM(self.label[label_idx])[0]
            if self.spoco:
                return raw_patch_transformed, raw_patch_transformed, label_patch_transformed
            return raw_patch_transformed, label_patch_transformed
        else:
            raw_patch = self.raw_transform(self.raw[raw_idx])
            if self.spoco:
                return raw_patch, raw_patch, raw_idx
            return raw_patch, raw_idx

    def __len__(self):
        return self.patch_count
