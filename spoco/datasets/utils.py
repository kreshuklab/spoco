import collections
from pathlib import Path

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
    elif args.ds_name == 'mitoem':
        ds_path = Path(args.ds_path)
        test_file = ds_path / 'val.h5'
        assert test_file.exists(), f'Test file {test_file} does not exist'
        assert len(args.patch_shape) == 3, 'Patch shape must be a 3D tuple'
        assert len(args.stride_shape) == 3, 'Stride shape must be a 3D tuple'
        assert args.patch_shape[0] == 1, 'Patch shape must have a depth of 1: only 2D patches are supported'
        assert args.stride_shape[0] == 1, 'Stride shape must have a depth of 1: only 2D patches are supported'
        test_dataset = VolumetricH5Dataset(test_file, phase='test', patch_shape=args.patch_shape,
                                           stride_shape=args.stride_shape, spoco=args.spoco)
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
