import argparse
import glob
import os
from concurrent import futures

import h5py
import numpy as np

from spoco.clustering.utils import cluster_ms, cluster_hdbscan, cluster_consistency
from spoco.utils import SUPPORTED_DATASETS

parser = argparse.ArgumentParser(description='Embedding clustering')
parser.add_argument('--ds-name', type=str, default='cvppp', choices=SUPPORTED_DATASETS,
                    help=f'Name of the dataset from: {SUPPORTED_DATASETS}')
parser.add_argument('--emb-dir', type=str, help='Path to embedding predictions directory', required=True)
parser.add_argument('--clustering', type=str, help='Clustering algorithm: ms/hdbscan/consistency', required=True)
parser.add_argument('--delta-var', type=float, help='delta_var param', default=0.5)
parser.add_argument('--min-size', type=int, help='HDBSCAN min_size param', default=50)
parser.add_argument('--output-dataset', type=str, help='H5 dataset name where segmentation will be saved',
                    default='segmentation')
parser.add_argument('--remove-largest', help='Set largest instance to 0-label', action='store_true')
parser.add_argument('--iou-threshold', type=float, help='IoU threshold for consistency clustering', required=False,
                    default=0.6)
parser.add_argument('--num-workers', type=int, help='Thread pool size', default=32)

args = parser.parse_args()


def cluster_image(pred_file, clustering, eps, min_size, output_dataset, remove_largest, iou_threshold):
    # load embeddings from H5
    embs = []
    with h5py.File(pred_file, 'r') as f:
        for k in ['embeddings1', 'embeddings2']:
            if k in f:
                embs.append(f[k][:])

    assert clustering in ('ms', 'hdbscan', 'consistency'), \
        f"Unsupported clustering algorithm '{clustering}'. Supported values: ms, hdbscan, consistency"

    clustering_results = []
    if clustering == 'ms':
        for emb in embs:
            clusters = cluster_ms(emb, bandwidth=eps)
            clustering_results.append(clusters)

    elif clustering == 'hdbscan':
        for emb in embs:
            clusters = cluster_hdbscan(emb, min_size=min_size, eps=eps)
            clustering_results.append(clusters)
    else:
        assert len(embs) == 2
        clusters = cluster_consistency(embs[0], embs[1], eps, iou_threshold)
        clustering_results.append(clusters)

    if remove_largest:
        for clusters in clustering_results:
            ids, counts = np.unique(clusters, return_counts=True)
            clusters[ids[np.argmax(counts)] == clusters] = 0

    # save results in the H5
    with h5py.File(pred_file, 'r+') as f:
        for i, clusters in enumerate(clustering_results):
            if len(clustering_results) > 1:
                out_ds = f'{output_dataset}_{i+1}'
            else:
                out_ds = output_dataset

            if out_ds in f:
                del f[out_ds]

            f.create_dataset(out_ds, data=clusters.astype('uint32'), compression='gzip')

    print(f'{pred_file} processed!')


def cluster_images(args):
    with futures.ProcessPoolExecutor(args.num_workers) as executor:
        tasks = []
        # load h5 predictions files
        for pred_file in glob.glob(os.path.join(args.emb_dir, '*predictions.h5')):
            _, filename = os.path.split(pred_file)

            print(f'Processing {filename}...')
            task = executor.submit(cluster_image,
                                   pred_file,
                                   args.clustering,
                                   args.delta_var,
                                   args.min_size,
                                   args.output_dataset,
                                   args.remove_largest,
                                   args.iou_threshold)
            tasks.append(task)

        results = [t.result() for t in tasks]

    print()
    results = np.array(results)
    print(f'Dice Score: {np.mean(results)}')


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.emb_dir)
    print(f"Clustering embeddings from '{args.emb_dir}' using '{args.clustering}'")

    if args.ds_name == 'cvppp':
        cluster_images(args)
    elif args.ds_name == 'dsb':
        cluster_images(args)
    elif args.ds_name == 'ovules':
        # TODO
        pass
    elif args.ds_name == 'stem':
        # TODO
        pass
    elif args.ds_name == 'mitoem':
        # TODO
        pass
    else:
        raise ValueError(f'Unsupported dataset name: {args.ds_name}')
