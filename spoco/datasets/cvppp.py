import os
import random

import numpy as np
import torch
from PIL import Image
from PIL import ImageFile

from spoco.transforms import RgbToLabel, Relabel, GaussianBlur, ImgNormalize, LabelToTensor

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms


def cvppp_sample_instances(pil_img, instance_ratio, random_state, ignore_labels=(0,)):
    # convert PIL image to np.array
    label_img = np.array(pil_img)

    # convert RGB to int
    label = RgbToLabel()(label_img)
    # relabel
    label = Relabel(run_cc=False)(label)

    unique = np.unique(label)
    for il in ignore_labels:
        unique = np.setdiff1d(unique, il)

    # shuffle labels
    random_state.shuffle(unique)
    # pick instance_ratio objects
    num_objects = round(instance_ratio * len(unique))
    if num_objects == 0:
        # if there are no objects left, just return an empty patch
        return np.zeros_like(label_img)

    # sample the labels
    sampled_instances = unique[:num_objects]

    mask = np.zeros_like(label)
    # keep only the sampled_instances
    for si in sampled_instances:
        mask[label == si] = 1
    # mask each channel
    mask = mask.astype('uint8')
    mask = np.stack([mask] * 3, axis=2)
    label_img = label_img * mask

    return Image.fromarray(label_img)


class DimExtender:
    def __init__(self, should_extend):
        self.should_extend = should_extend

    def __call__(self, m):
        if self.should_extend:
            return m.unsqueeze(0)
        return m


DEFAULT_RAW_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(448, scale=(0.7, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            GaussianBlur([.1, 2.])
        ], p=0.5),
        transforms.ToTensor(),
        ImgNormalize()
    ]
)

# used with SPOCO embedding trainer
BASE_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(448, scale=(0.7, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ]
)

EXTENDED_TRANSFORM = transforms.Compose(
    [
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            GaussianBlur([.1, 2.])
        ], p=0.5),
        transforms.ToTensor(),
        ImgNormalize()
    ]
)

SPOCO_TEST = transforms.Compose(
    [
        transforms.Resize(size=(448, 448)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        GaussianBlur([.1, 2.]),
        transforms.ToTensor(),
        ImgNormalize()
    ]
)


class CVPPP2017Dataset:
    def __init__(self, root_dir, phase, spoco=False, instance_ratio=None, seed=None):
        assert os.path.isdir(root_dir), f'{root_dir} is not a directory'
        assert phase in ['train', 'val', 'test']

        if spoco:
            raw_transform = BASE_TRANSFORM
        else:
            raw_transform = DEFAULT_RAW_TRANSFORM

        self.phase = phase
        root_dir = os.path.join(root_dir, phase, 'A1')
        self.images, self.paths = self._load_files(root_dir, suffix='rgb')
        assert len(self.images) > 0

        self.instance_ratio = instance_ratio
        self.raw_transform = raw_transform
        self.spoco = spoco

        self.train_label_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(448, scale=(0.7, 1.), interpolation=0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RgbToLabel(),
                Relabel(run_cc=False),
                LabelToTensor(False),
            ]
        )

        self.val_label_transform = transforms.Compose(
            [
                transforms.Resize(size=(448, 448), interpolation=Image.NEAREST),
                RgbToLabel(),
                Relabel(run_cc=False),
                LabelToTensor(False),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(size=(448, 448)),
                transforms.ToTensor(),
                ImgNormalize()
            ]
        )

        if phase == 'train':
            # load labeled images
            self.masks, _ = self._load_files(root_dir, suffix='label')
            # training with sparse object supervision
            if self.instance_ratio is not None and phase == 'train':
                assert 0 < self.instance_ratio <= 1
                rs = np.random.RandomState(seed)
                self.masks = [cvppp_sample_instances(m, self.instance_ratio, rs) for m in self.masks]

            assert len(self.images) == len(self.masks)
        elif phase == 'val':
            # load labeled images
            self.masks, _ = self._load_files(root_dir, suffix='label')
            assert len(self.images) == len(self.masks)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img = self.images[idx]
        if self.phase == 'train':
            mask = self.masks[idx]
            seed = np.random.randint(np.iinfo('int32').max)
            random.seed(seed)
            # make it work with torch vision v0.8.0+
            torch.manual_seed(seed)
            img = self.raw_transform(img)
            random.seed(seed)
            # make it work with torch vision v0.8.0+
            torch.manual_seed(seed)
            mask = self.train_label_transform(mask)
            if self.spoco:
                img1 = EXTENDED_TRANSFORM(img)
                img2 = EXTENDED_TRANSFORM(img)
                return img1, img2, mask
            else:
                return img, mask
        elif self.phase == 'val':
            mask = self.masks[idx]
            seed = np.random.randint(np.iinfo('int32').max)
            random.seed(seed)
            img = self.test_transform(img)
            random.seed(seed)
            mask = self.val_label_transform(mask)
            if self.spoco:
                return img, img, mask
            return img, mask
        else:
            if self.spoco:
                img1 = self.test_transform(img)
                img2 = SPOCO_TEST(img)
                return img1, img2, self.paths[idx]
            return self.test_transform(img), self.paths[idx]

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _load_files(dir, suffix):
        # we only load raw or label images
        assert suffix in ['rgb', 'label', 'fg']
        files_data = []
        paths = []
        for file in sorted(os.listdir(dir)):
            base = os.path.splitext(file)[0]
            if base.endswith(suffix):
                path = os.path.join(dir, file)
                # load image
                img = Image.open(path)
                if suffix in ['rgb', 'label']:
                    img = img.convert('RGB')
                # save image and path
                files_data.append(img)
                paths.append(path)

        return files_data, paths
