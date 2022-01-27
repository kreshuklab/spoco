import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from spoco.transforms import ImgNormalize, Relabel, GaussianBlur

VALID_CLASSES = [
    7,
    8,
    11,
    12,
    13,
    17,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    31,
    32,
    33,
]
CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle"
]

CLASS_MAP = dict(zip(CLASS_NAMES, VALID_CLASSES))

BASE_RAW_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop((384, 768), scale=(0.5, 2.)),
        transforms.RandomHorizontalFlip()
    ]
)

TEST_RAW_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(size=(384, 768)),
        transforms.ToTensor(),
        ImgNormalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])

    ]
)

LABEL_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop((384, 768), scale=(0.5, 2.), interpolation=0),
        transforms.RandomHorizontalFlip(),
        Relabel(run_cc=False),
        transforms.ToTensor()
    ]
)

TEST_LABEL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(size=(384, 768), interpolation=Image.NEAREST),
        Relabel(run_cc=False),
        transforms.ToTensor()
    ]
)

EXTENDED_TRANSFORM = transforms.Compose(
    [
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur([.1, 2.]),
        transforms.ToTensor(),
        ImgNormalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ]
)


class CityscapesDataset:
    def __init__(self, root_dir, phase, class_name, instance_ratio=None, spoco=False):
        assert phase in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.phase = phase
        self.class_name = class_name
        self.class_id = CLASS_MAP[class_name]
        self.spoco = spoco
        self.img_normalize = transforms.Compose([
            transforms.ToTensor(),
            ImgNormalize()
        ])

        self.images_base = os.path.join(root_dir, 'leftImg8bit', phase)
        self.annotations_base = os.path.join(root_dir, 'gtFine', phase)

        if instance_ratio is None or phase in ['val', 'test']:
            self.instance_ratio = '1.0'
        else:
            self.instance_ratio = str(instance_ratio)

        # we only load files containing a given class, since we train for each class separately
        suffix = self.instance_ratio
        file_list = os.path.join(self.images_base, f'{self.class_name}_{suffix}.txt')
        with open(file_list) as f:
            self.raw_files = f.read().splitlines()

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img_path = self.raw_files[idx]
        img = Image.open(img_path)

        if self.phase == 'train':
            inst_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-2],
                self.class_name,
                self.instance_ratio,
                os.path.basename(img_path)[:-15] + "gtFine_instanceIds.png"
            )
            mask = Image.open(inst_path)

            seed = np.random.randint(np.iinfo('int32').max)
            random.seed(seed)
            torch.manual_seed(seed)
            img = BASE_RAW_TRANSFORM(img)
            random.seed(seed)
            torch.manual_seed(seed)
            mask = LABEL_TRANSFORM(mask)[0]
            if self.spoco:
                img2 = EXTENDED_TRANSFORM(img)
                # normalize img
                img = self.img_normalize(img)
                return img, img2, mask
            else:
                # normalize img
                img = self.img_normalize(img)
                return img, mask
        elif self.phase == 'val':
            inst_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-2],
                self.class_name,
                '1.0',
                os.path.basename(img_path)[:-15] + "gtFine_instanceIds.png"
            )
            mask = Image.open(inst_path)

            img = TEST_RAW_TRANSFORM(img)
            mask = TEST_LABEL_TRANSFORM(mask)[0]
            if self.spoco:
                return img, img, mask
            return img, mask
        else:
            img = TEST_RAW_TRANSFORM(img)
            if self.spoco:
                return img, img, self.raw_files[idx]
            return img, self.raw_files[idx]

    def __len__(self):
        return len(self.raw_files)
