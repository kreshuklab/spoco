import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from spoco.transforms import Relabel, GaussianBlurNp, Standardize, RandomFlip

LABEL_TRANSFORM = transforms.Compose(
    [
        RandomFlip(),
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
        instance_ratio (float): ratio of instances to be used for self-training
        spoco (bool): if True, the dataset is used for spoco training
    """

    def __init__(self, file_path, phase, patch_shape, stride_shape,
                 raw_internal_path='raw', label_internal_path='label',
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

        with h5py.File(file_path, 'r') as f:
            raw = f[raw_internal_path][:]
            raw_mean, raw_std = raw.mean(), raw.std()

        self.train_raw_transform = transforms.Compose(
            [
                RandomFlip(),
                Standardize(mean=raw_mean, std=raw_std),
                transforms.ToTensor()
            ]
        )

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

        raw_img = self.raw[raw_idx][0]
        if self.phase == 'train':
            seed = np.random.randint(np.iinfo('int32').max)
            random.seed(seed)
            torch.manual_seed(seed)
            raw_patch_transformed = self.train_raw_transform(raw_img)
            random.seed(seed)
            torch.manual_seed(seed)
            label_idx = self.label_slices[idx]
            label_img = self.label[label_idx][0]
            label_patch_transformed = LABEL_TRANSFORM(label_img)
            # remove channel dim
            label_patch_transformed = label_patch_transformed[0]
            if self.spoco:
                raw_patch_transformed1 = EXTENDED_TRANSFORM(raw_patch_transformed)
                return raw_patch_transformed, raw_patch_transformed1, label_patch_transformed
            else:
                return raw_patch_transformed, label_patch_transformed
        elif self.phase == 'val':
            raw_patch_transformed = self.raw_transform(raw_img)
            label_idx = self.label_slices[idx]
            label_img = self.label[label_idx][0]
            label_patch_transformed = TEST_LABEL_TRANSFORM(label_img)
            # remove channel dim
            label_patch_transformed = label_patch_transformed[0]
            if self.spoco:
                return raw_patch_transformed, raw_patch_transformed, label_patch_transformed
            return raw_patch_transformed, label_patch_transformed
        else:
            raw_patch = self.raw_transform(raw_img)
            if self.spoco:
                return raw_patch, raw_patch, raw_idx
            return raw_patch, raw_idx

    def __len__(self):
        return self.patch_count


class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label ndarray based on the patch and stride shape.

    Args:
        raw_dataset (ndarray): raw data
        label_dataset (ndarray): ground truth labels
        patch_shape (tuple): the shape of the patch DxHxW
        stride_shape (tuple): the shape of the stride DxHxW
    """

    def __init__(self, raw_dataset, label_dataset, patch_shape, stride_shape):
        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)

        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            self._label_slices = None
        else:
            self._label_slices = self._build_slices(label_dataset, patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices), 'Raw and label slices must have the same length'

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x),
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'


class FilterSliceBuilder(SliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label
    """

    def __init__(self, raw_dataset, label_dataset, patch_shape, stride_shape, ignore_index=None,
                 threshold=0.05, slack_acceptance=0.01):
        super().__init__(raw_dataset, label_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            label_idx = raw_label_idx[1]
            patch = label_dataset[label_idx]
            if ignore_index is not None:
                patch = np.copy(patch)
                patch[patch == ignore_index] = 0
            non_ignore_counts = np.count_nonzero(patch != 0)
            non_ignore_counts = non_ignore_counts / patch.size
            return non_ignore_counts > threshold or rand_state.rand() < slack_acceptance

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)
