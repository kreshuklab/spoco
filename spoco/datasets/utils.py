import collections
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from spoco.datasets.cityscapes import CityscapesDataset
from spoco.datasets.cvppp import CVPPP2017Dataset
from spoco.datasets.volumetric import VolumetricH5Dataset


def create_train_val_loaders(args):
    """
    Creates train and validation data loaders.

    Args:
        args: command line arguments

    Returns:
        (train_loader, val_loader): train and validation data loaders
    """
    if args.ds_name == 'cvppp':
        train_dataset = CVPPP2017Dataset(args.ds_path, phase='train', spoco=args.spoco,
                                         instance_ratio=args.instance_ratio, seed=args.manual_seed)
        val_dataset = CVPPP2017Dataset(args.ds_path, 'val', spoco=args.spoco)
    elif args.ds_name == 'cityscapes':
        train_dataset = CityscapesDataset(args.ds_path, phase='train', class_name=args.things_class, spoco=args.spoco,
                                          instance_ratio=args.instance_ratio)
        val_dataset = CityscapesDataset(args.ds_path, phase='val', class_name=args.things_class, spoco=args.spoco)
    elif args.ds_name == 'mitoem':
        ds_path = Path(args.ds_path)
        train_file = ds_path / 'train.h5'
        val_file = ds_path / 'val.h5'
        assert train_file.exists(), f'Training file {train_file} does not exist'
        assert val_file.exists(), f'Validation file {val_file} does not exist'
        assert len(args.patch_shape) == 3, 'Patch shape must be a 3D tuple'
        assert len(args.stride_shape) == 3, 'Stride shape must be a 3D tuple'
        assert args.patch_shape[0] == 1, 'Patch shape must have a depth of 1: only 2D patches are supported'
        assert args.stride_shape[0] == 1, 'Stride shape must have a depth of 1: only 2D patches are supported'
        train_dataset = VolumetricH5Dataset(train_file, phase='train', patch_shape=args.patch_shape,
                                            stride_shape=args.stride_shape, spoco=args.spoco,
                                            instance_ratio=args.instance_ratio)
        val_dataset = VolumetricH5Dataset(val_file, phase='val', patch_shape=args.patch_shape,
                                          stride_shape=args.stride_shape, spoco=args.spoco)
    else:
        raise RuntimeError(f'Unsupported dataset: {args.ds_name}')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    # shuffling should be done in the Sampler
    return [DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=False),
            DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, sampler=val_sampler, drop_last=False)]


def create_test_loader(args):
    """
    Creates test set data loader.
    Args:
        args: command line arguments

    Returns:
        test_loader: test set data loader
    """
    if args.ds_name == 'cvppp':
        test_dataset = CVPPP2017Dataset(args.ds_path, phase='test', spoco=args.spoco)
    elif args.ds_name == 'cityscapes':
        test_dataset = CityscapesDataset(args.ds_path, phase='test', class_name=None, spoco=args.spoco)
    else:
        raise RuntimeError(f'Unsupported dataset {args.ds_name}')

    return DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                      collate_fn=default_prediction_collate)


def default_prediction_collate(batch):
    """
    Forms a mini-batch of (images, paths) during test time for the DSB-like datasets.
    """
    error_msg = "batch must contain tensors or str; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], collections.Sequence):
        # transpose tuples, i.e. [[1, 2], ['a', 'b']] to be [[1, 'a'], [2, 'b']]
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def h5_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [h5_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label ndarray based on the patch and stride shape.

    Args:
        raw_dataset (ndarray): raw data
        label_dataset (ndarray): ground truth labels
        patch_shape (tuple): the shape of the patch DxHxW
        stride_shape (tuple): the shape of the stride DxHxW
    """

    def __init__(self, raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape):
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
