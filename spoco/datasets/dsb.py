import os

import imageio
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from spoco.transforms import AdditiveGaussianNoise, ToTensor, AdditivePoissonNoise, CropToFixed, Relabel, \
    PercentileNormalizer, Standardize, RandomFlip, RandomRotate90, RandomRotate, ElasticDeformation

EXTENDED_TRANSFORM = Compose([
    AdditiveGaussianNoise(np.random.RandomState(), scale=(0.0, 0.5), execution_probability=0.5),
    AdditivePoissonNoise(np.random.RandomState(), lam=(0.0, 0.5), execution_probability=0.5),
    ToTensor(expand_dims=False)
])

VAL_LABEL_TRANSFORM = Compose([
    CropToFixed(np.random.RandomState(), size=(256, 256), centered=True),
    Relabel(run_cc=False),
    ToTensor(expand_dims=False, dtype='int64')
])

TEST_TRANSFORM = Compose([
    CropToFixed(np.random.RandomState(), size=(256, 256), centered=True, channelwise=True),
    PercentileNormalizer(pmin=1, pmax=99.8, channelwise=True),
    Standardize(channelwise=True),
    ToTensor(expand_dims=False)
])

TEST_TRANSFORM_AUG = Compose([
    CropToFixed(np.random.RandomState(), size=(256, 256), centered=True, channelwise=True),
    PercentileNormalizer(pmin=1, pmax=99.8, channelwise=True),
    Standardize(channelwise=True),
    AdditiveGaussianNoise(np.random.RandomState(), scale=(0.0, 0.5), execution_probability=0.5),
    AdditivePoissonNoise(np.random.RandomState(), lam=(0.0, 0.5), execution_probability=0.5),
    ToTensor(expand_dims=False)
])


class DSB2018Dataset(Dataset):
    def __init__(self, root_dir, phase, instance_ratio=None, random_seed=None):
        assert phase in ['train', 'val', 'test']
        root_dir = os.path.join(root_dir, phase)
        assert os.path.isdir(root_dir), f'{root_dir} is not a directory'

        self.phase = phase

        if random_seed is None:
            random_seed = np.random.randint(np.iinfo('int32').max)

        rs_raw = np.random.RandomState(random_seed)
        rs_label = np.random.RandomState(random_seed)

        self.raw_transform = Compose([
            CropToFixed(rs_raw, size=(256, 256), channelwise=True),
            PercentileNormalizer(pmin=1, pmax=99.8, channelwise=True),
            Standardize(channelwise=True),
            RandomFlip(rs_raw, channelwise=True),
            RandomRotate90(rs_raw, channelwise=True),
            RandomRotate(rs_raw, angle_spectrum=45, mode='reflect', order=3, channelwise=True),
            ElasticDeformation(rs_raw, spline_order=3, execution_probability=0.2, channelwise=True),
        ])

        self.masks_transform = Compose([
            CropToFixed(rs_label, size=(256, 256)),
            RandomFlip(rs_label),
            RandomRotate90(rs_label),
            RandomRotate(rs_label, angle_spectrum=45, mode='reflect', order=0),
            ElasticDeformation(rs_label, spline_order=0, execution_probability=0.2),
            Relabel(run_cc=False),
            ToTensor(expand_dims=False, dtype='int64')
        ])

        # load raw images
        images_dir = os.path.join(root_dir, 'images')
        assert os.path.isdir(images_dir)
        self.images, self.paths = self._load_files(images_dir, is_raw=True)
        self.file_path = images_dir
        self.instance_ratio = instance_ratio

        if phase != 'test':
            # load labeled images
            masks_dir = os.path.join(root_dir, 'masks')
            assert os.path.isdir(masks_dir)
            self.masks, _ = self._load_files(masks_dir, is_raw=False)
            # prepare for training with sparse object supervision (allow sparse objects only in training phase)
            if self.instance_ratio is not None and phase == 'train':
                assert 0 < self.instance_ratio <= 1
                rs = np.random.RandomState(random_seed)
                self.masks = [sample_instances(m, self.instance_ratio, rs) for m in self.masks]
            assert len(self.images) == len(self.masks)
            # load label images transformer
        else:
            self.masks = None

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img = self.images[idx]
        if self.phase == 'train':
            mask = self.masks[idx]
            # apply base transforms to raw and masks
            raw_transformed = self.raw_transform(img)
            mask_transformed = self.masks_transform(mask)

            # apply geometry preserving augmentations to raw
            raw1 = EXTENDED_TRANSFORM(raw_transformed)
            raw2 = EXTENDED_TRANSFORM(raw_transformed)

            return raw1, raw2, mask_transformed
        elif self.phase == 'val':
            mask = self.masks[idx]
            mask = VAL_LABEL_TRANSFORM(mask)
            img1 = TEST_TRANSFORM(img)
            img2 = TEST_TRANSFORM_AUG(img)
            return img1, img2, mask
        else:
            return TEST_TRANSFORM(img), TEST_TRANSFORM_AUG(img), self.paths[idx]

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _load_files(root_dir, is_raw):
        files_data = []
        paths = []
        for file in os.listdir(root_dir):
            path = os.path.join(root_dir, file)
            img = np.asarray(imageio.imread(path))

            if is_raw:
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=2)

                img = np.transpose(img, (2, 0, 1))

            files_data.append(img)
            paths.append(path)

        return files_data, paths


def sample_instances(label_img, instance_ratio, random_state, ignore_labels=(0,)):
    """
    Given the labelled volume `label_img`, this function takes a random subset of object instances specified by `instance_ratio`
    and zeros out the remaining labels.

    Args:
        label_img(nd.array): labelled image
        instance_ratio(float): a number from (0, 1]
        random_state: RNG state
        ignore_labels: labels to be ignored during sampling

    Returns:
         labelled volume of the same size as `label_img` with a random subset of object instances.
    """
    unique = np.unique(label_img)
    for il in ignore_labels:
        unique = np.setdiff1d(unique, il)

    # shuffle labels
    random_state.shuffle(unique)
    # pick instance_ratio objects
    num_objects = round(instance_ratio * len(unique))
    if num_objects == 0:
        # if there are no objects left, just return an empty patch
        return np.zeros_like(label_img)

    if instance_ratio < 0.5:
        # sample the labels
        sampled_instances = unique[:num_objects]
        result = np.zeros_like(label_img)
        # keep only the sampled_instances
        for si in sampled_instances:
            result[label_img == si] = si
    else:
        not_sampled_instances = unique[num_objects:]
        result = label_img.copy()
        # discard only the not sampled instances
        for nsi in not_sampled_instances:
            result[label_img == nsi] = 0

    return result
