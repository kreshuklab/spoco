import argparse
import os

import imageio
import numpy as np
from tqdm import tqdm

from spoco.datasets.cityscapes import CLASS_NAMES, CLASS_MAP
from spoco.transforms import Relabel

THINGS = ['bicycle', 'bus', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck']

THINGS_IDS = [CLASS_MAP[cn] for cn in THINGS]


def traverse_dir(root_dir, suffix=""):
    files = []
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(suffix):
                files.append(os.path.join(root, filename))
    return files


def load_labels(raw_files, annotations_base, class_id):
    labeled_imgs = []

    # assign unique instance ids across images
    max_id = 0
    for raw_img in tqdm(raw_files):
        lbl_path = os.path.join(
            annotations_base,
            raw_img.split(os.sep)[-2],
            os.path.basename(raw_img)[:-15] + "gtFine_labelIds.png",
        )
        inst_path = os.path.join(
            annotations_base,
            raw_img.split(os.sep)[-2],
            os.path.basename(raw_img)[:-15] + "gtFine_instanceIds.png",
        )
        lbl_img = np.array(imageio.imread(lbl_path))
        unique = np.unique(lbl_img)
        if class_id is None or class_id in unique:
            inst_img = np.array(imageio.imread(inst_path))
            inst_img = inst_img.astype('uint32')
            if class_id is None:
                # leave 'things' and remove 'stuff'
                tmp_img = np.zeros_like(inst_img)
                for cid in THINGS_IDS:
                    tmp_img[lbl_img == cid] = inst_img[lbl_img == cid]
                inst_img = tmp_img
            else:
                # leave only the class_id objects
                inst_img[lbl_img != class_id] = 0
            # relabel
            _, unique_ids = np.unique(inst_img, return_inverse=True)
            inst_img = unique_ids.reshape(inst_img.shape)
            # make labels unique across images
            inst_img[inst_img > 0] += max_id
            if inst_img.max() > 0:
                # update max_id
                max_id = inst_img.max()
            labeled_imgs.append(inst_img)
        else:
            # image does not contain objects of the given class
            labeled_imgs.append(np.zeros_like(lbl_img))

    return labeled_imgs, max_id


def cityscapes_sample_instances(raw_files, labeled_imgs, max_id, instance_ratio, rs):
    ids = list(range(1, max_id + 1))
    rs.shuffle(ids)
    # pick instance_ratio objects
    num_objects = round(instance_ratio * len(ids))
    if num_objects == 0:
        print('Not enough objects left for training!')
        return None, None

    print(f'Sampled {num_objects} out of {len(ids)} objects. Instance ratio: {instance_ratio}')

    # create a set of object ids left for training
    sampled_ids = set(ids[:num_objects])

    new_raw_files = []
    new_labeled_imgs = []
    for raw_file, labeled_img in tqdm(zip(raw_files, labeled_imgs)):
        # skip zero label
        unique_ids = np.unique(labeled_img)[1:]
        new_labeled_img = np.copy(labeled_img)
        for id in unique_ids:
            if not id in sampled_ids:
                new_labeled_img[new_labeled_img == id] = 0

        # save only non-empty images
        if new_labeled_img.sum() > 0:
            new_raw_files.append(raw_file)
            new_labeled_imgs.append(new_labeled_img)

    print(f'Number of images left for training: {len(new_raw_files)}')
    return new_raw_files, new_labeled_imgs


def save_labeled_images(raw_files, labeled_imgs, annotations_base, class_name, instance_ratio):
    rl = Relabel(run_cc=False)
    for raw_file, labeled_img in zip(raw_files, labeled_imgs):
        inst_path = os.path.join(
            annotations_base,
            raw_file.split(os.sep)[-2],
            class_name,
            instance_ratio,
            os.path.basename(raw_file)[:-15] + "gtFine_instanceIds.png",
        )
        inst_dir = os.path.split(inst_path)[0]
        if not os.path.exists(inst_dir):
            os.makedirs(inst_dir)
        print(f'Saving {inst_path}')
        labeled_img = rl(labeled_img)
        imageio.imwrite(inst_path, labeled_img.astype('uint8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, help='path to base dir', required=True)
    parser.add_argument('--class_names', nargs="+", type=str, help='class names', default=None)

    args = parser.parse_args()

    if args.class_names is not None:
        for class_name in args.class_names:
            assert class_name in CLASS_NAMES
        class_names = args.class_names
    else:
        class_names = [None]

    for phase in ['train', 'val', 'test']:
        annotations_base = os.path.join(args.base_dir, 'gtFine', phase)
        images_base = os.path.join(args.base_dir, 'leftImg8bit', phase)
        raw_files = traverse_dir(images_base, suffix='.png')
        for class_name in class_names:
            class_id = CLASS_MAP.get(class_name)
            print(f'Loading annotations from {annotations_base}, class: {class_name}, class_id: {class_id}')
            labeled_imgs, max_id = load_labels(raw_files, annotations_base, class_id)
            for instance_ratio in ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']:
                # skip validation images sampling
                if phase in ['val', 'test'] and instance_ratio != '1.0':
                    continue

                ir = float(instance_ratio)
                rs = np.random.RandomState(47)
                print(f'Sampling {ir * 100}% of instances of class: {class_name}')

                sampled_raw_files, sampled_labeled_imgs = cityscapes_sample_instances(raw_files, labeled_imgs, max_id,
                                                                                      ir, rs)
                if sampled_raw_files is None:
                    continue

                # save raw files
                if class_name is None:
                    class_name = 'all'
                file_list = os.path.join(images_base, f'{class_name}_{instance_ratio}.txt')
                with open(file_list, mode='wt') as f:
                    f.write('\n'.join(sampled_raw_files))

                save_labeled_images(sampled_raw_files, sampled_labeled_imgs, annotations_base, class_name,
                                    instance_ratio)
