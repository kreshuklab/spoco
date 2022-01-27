import collections

import torch
from torch.utils.data import DataLoader

from spoco.datasets.cityscapes import CityscapesDataset
from spoco.datasets.cvppp import CVPPP2017Dataset


def create_train_val_loaders(args):
    if args.ds_name == 'cvppp':
        train_dataset = CVPPP2017Dataset(args.ds_path, phase='train', spoco=args.spoco,
                                         instance_ratio=args.instance_ratio, seed=args.manual_seed)
        val_dataset = CVPPP2017Dataset(args.ds_path, 'val', spoco=args.spoco)
    elif args.ds_name == 'cityscapes':
        train_dataset = CityscapesDataset(args.ds_path, phase='train', class_name=args.things_class, spoco=args.spoco,
                                          instance_ratio=args.instance_ratio)
        val_dataset = CityscapesDataset(args.ds_path, phase='val', class_name=args.things_class, spoco=args.spoco)
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
    if args.ds_name == 'cvppp':
        test_dataset = CVPPP2017Dataset(args.ds_path, 'test')
    elif args.ds_name == 'cityscapes':
        pass
    else:
        # TODO: add remaining dataset
        raise RuntimeError(f'Unsupported dataset {args.ds_name}')

    if args.ds_name in ('cvppp', 'cityscapes'):
        collate_fn = dsb_prediction_collate
    else:
        collate_fn = default_prediction_collate

    return DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)


def dsb_prediction_collate(batch):
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
        return [dsb_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def default_prediction_collate(batch):
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
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))
