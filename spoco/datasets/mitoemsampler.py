import argparse
from pathlib import Path

import h5py
import numpy as np


def mitoem_sample_instances(label, instance_ratio, random_state):
    label_img = np.copy(label)
    unique_ids = np.unique(label)[1:]
    rs.shuffle(unique_ids)
    # pick instance_ratio objects
    num_objects = round(instance_ratio * len(unique_ids))
    assert num_objects > 0, 'No objects to sample'
    print(f'Sampled {num_objects} out of {len(unique_ids)} objects. Instance ratio: {instance_ratio}')
    # create a set of object ids left for training
    sampled_ids = set(unique_ids[:num_objects])
    for id in unique_ids:
        if id not in sampled_ids:
            label_img[label_img == id] = 0
    return label_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='MitoEM dir containing train.h5 and val.h5 files',
                        required=True)
    parser.add_argument('--instance_ratios', nargs="+", type=float,
                        help='fraction of ground truth objects to sample.', required=True)

    args = parser.parse_args()

    # load label dataset from the train.h5 file
    train_file = Path(args.dataset_dir) / 'train.h5'
    assert train_file.exists(), f'{train_file} does not exist'

    with h5py.File(train_file, 'r+') as f:
        label = f['label'][:]

        for instance_ratio in args.instance_ratios:
            assert 0.0 <= instance_ratio <= 1.0, 'Instance ratio must be in [0, 1]'

            ir = float(instance_ratio)
            rs = np.random.RandomState(47)
            print(f'Sampling {ir * 100}% of mitoEM instances')

            label_sampled = mitoem_sample_instances(label, ir, rs)

            # save the sampled label dataset
            f.create_dataset(f'label_{instance_ratio}', data=label_sampled, compression='gzip')
