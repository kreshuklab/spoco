import argparse
import glob
import os
from concurrent import futures

import h5py
import imageio
import numpy as np
from PIL import Image
from skimage.transform import resize
from torchvision.transforms import transforms

from spoco.cluster import cluster_ms, cluster_hdbscan, cluster_consistency, cluster_ms_plus
from spoco.datasets.cityscapes import CLASS_MAP
from spoco.metrics import AveragePrecision, symmetric_best_dice
from spoco.transforms import RgbToLabel, Relabel
from spoco.utils import SUPPORTED_DATASETS

parser = argparse.ArgumentParser(description='Embedding clustering')
parser.add_argument('--ds-name', required=True, type=str, choices=SUPPORTED_DATASETS,
                    help=f'Name of the dataset from: {SUPPORTED_DATASETS}')
parser.add_argument('--gt-dir', type=str, default=None,
                    help=f'Path to the ground truth directory. If provided the segmentation scores will be computed.')
parser.add_argument('--emb-dir', type=str, help='Path to embedding predictions directory', required=True)
parser.add_argument('--things-class', type=str, help='Cityscapes semantic class', default=None)
parser.add_argument('--sem-dir', type=str, default=None,
                    help='Path to semantic segmentation predictions directory')
parser.add_argument('--clustering', type=str, help='Clustering algorithm: ms/msplus/hdbscan/consistency', required=True)
parser.add_argument('--delta-var', type=float, help='Pull force hinge', default=0.5)
parser.add_argument('--delta-dist', type=float, help='Push force hinge', default=2.0)
parser.add_argument('--min-size', type=int, help='HDBSCAN min_size param', default=50)
parser.add_argument('--output-dataset', type=str, help='H5 dataset name where segmentation will be saved',
                    default='segmentation')
parser.add_argument('--remove-largest', help='Set largest instance to 0-label', action='store_true')
parser.add_argument('--iou-threshold', type=float, help='IoU threshold for consistency clustering', required=False,
                    default=0.6)
parser.add_argument('--num-workers', type=int, help='Thread pool size', default=32)

args = parser.parse_args()

DEEPLAB_CLASS_MAP = {
    'person': 300,
    'rider': 255,
    'car': 142,
    'truck': 70,
    'bus': 160,
    'trailer': 110,
    'train': 180,
    'motorcycle': 230,
    'bicycle': 162
}


def load_cityscapes_ground_truth(base_dir, class_id, filename):
    city = filename.split('_')[0]
    city_dir = os.path.join(base_dir, city)

    lbl_path = os.path.join(
        city_dir,
        filename[:-15] + "gtFine_labelIds.png",
    )
    inst_path = os.path.join(
        city_dir,
        filename[:-15] + "gtFine_instanceIds.png",
    )

    lbl_img = np.array(imageio.imread(lbl_path))
    unique = np.unique(lbl_img)
    if class_id in unique:
        inst_img = np.array(imageio.imread(inst_path))
        inst_img = inst_img.astype('uint32')

        # leave only the class_id objects
        inst_img[lbl_img != class_id] = 0
        # relabel
        _, unique_ids = np.unique(inst_img, return_inverse=True)
        inst_img = unique_ids.reshape(inst_img.shape)
        # resize
        inst_img = resize(inst_img, output_shape=(384, 768), order=0, preserve_range=True, anti_aliasing=False).astype(
            'int64')
        return inst_img

    return None


def process_cityscapes_sem_mask(sem_filepath, class_name, size=(384, 768)):
    t = transforms.Resize(size=size, interpolation=Image.NEAREST)
    img = Image.open(sem_filepath)
    # resize
    img = t(img)
    img = np.array(img)
    img = np.sum(img, axis=2)
    mask = np.zeros_like(img)
    mask[img == DEEPLAB_CLASS_MAP[class_name]] = 1
    return mask


def load_cityscapes_sem_mask(root_dir, pred_file, class_name, min_size):
    filename = os.path.basename(pred_file)
    sem_filepath = os.path.join(root_dir, filename[:-15] + '.png')
    print(f'Semantic file {sem_filepath}')
    semantic_mask = process_cityscapes_sem_mask(sem_filepath, class_name)
    # remove car reflection false positives
    semantic_mask[350:, :] = 0
    # skip images for which no clusters can be formed
    if semantic_mask.sum() < min_size:
        print(f'Skipping sem file {sem_filepath}')
        return None
    return semantic_mask


class AbstractClustering:
    def __init__(self, args):
        self.args = args

    def __call__(self, embs, pred_file):
        algorithm = self.args.clustering
        # load semantic mask if provided
        semantic_mask = self.load_semantic_mask(pred_file)
        if algorithm == 'ms':
            # use emb1 for clustering only
            clusters = cluster_ms(embs[0], bandwidth=self.args.delta_var, semantic_mask=semantic_mask)
        elif algorithm == 'msplus':
            if semantic_mask is None:
                return None

            # use emb1 for clustering only
            clusters = cluster_ms_plus(embs[0], bandwidth=self.args.delta_var, delta_dist=self.args.delta_dist,
                                       semantic_mask=semantic_mask)
        elif algorithm == 'hdbscan':
            # use emb1 for clustering only
            clusters = cluster_hdbscan(embs[0], min_size=self.args.min_size, eps=self.args.delta_var,
                                       semantic_mask=semantic_mask)
        else:
            assert len(embs) == 2
            clusters = cluster_consistency(embs[0], embs[1], bandwidth=self.args.delta_var,
                                           iou_threshold=self.args.iou_threshold, semantic_mask=None)

        if self.args.remove_largest:
            ids, counts = np.unique(clusters, return_counts=True)
            clusters[ids[np.argmax(counts)] == clusters] = 0

        # save results in the H5
        with h5py.File(pred_file, 'r+') as f:
            out_ds = self.args.output_dataset
            # override previous segmentation if exists
            if out_ds in f:
                del f[out_ds]
            print(f'Saving segmentation results to: {pred_file}:{out_ds}')
            f.create_dataset(out_ds, data=clusters.astype('uint32'), compression='gzip')

            # load ground truth if provided
            if self.args.gt_dir is not None:
                gt = self.load_groundtruth(pred_file)
                if gt is None:
                    return None
                # save gt into the prediction file
                gt_ds = 'gt'
                if self.args.things_class is not None:
                    gt_ds = 'gt_' + self.args.things_class
                if gt_ds in f:
                    del f[gt_ds]
                f.create_dataset(gt_ds, data=gt, compression='gzip')
                # return score
                return self.segmentation_score(clusters, gt)

        return None

    def load_groundtruth(self, pred_file):
        raise NotImplementedError

    def segmentation_score(self, clusters, gt):
        raise NotImplementedError

    def load_semantic_mask(self, pred_file):
        raise NotImplementedError


class CvpppClustering(AbstractClustering):
    def segmentation_score(self, clusters, gt):
        return symmetric_best_dice(gt, clusters)

    def _load_mask(self, pred_file, suffix):
        filename = os.path.basename(pred_file)
        prefix = filename.split('_')[0]
        label_file = os.path.join(self.args.gt_dir, prefix + suffix)
        img = Image.open(label_file).convert('RGB')

        label_transform = transforms.Compose([
            transforms.Resize(size=(448, 448), interpolation=Image.NEAREST),
            RgbToLabel(),
            Relabel(run_cc=False)
        ])
        img = label_transform(img)
        return img

    def load_semantic_mask(self, pred_file):
        if self.args.sem_dir is not None:
            return self._load_mask(pred_file, '_fg.png')
        return None

    def load_groundtruth(self, pred_file):
        return self._load_mask(pred_file, '_label.png')


class CityscapesClustering(AbstractClustering):
    def segmentation_score(self, clusters, gt):
        ap = AveragePrecision(iou=0.5)
        return ap(clusters, gt)

    def load_semantic_mask(self, pred_file):
        return load_cityscapes_sem_mask(self.args.sem_dir, pred_file, self.args.things_class,
                                        self.args.min_size)

    def load_groundtruth(self, pred_file):
        filename = os.path.basename(pred_file)
        class_id = CLASS_MAP.get(self.args.things_class)
        return load_cityscapes_ground_truth(self.args.gt_dir, class_id, filename[:-15] + '.png')


def cluster_image(pred_file, args):
    # load embeddings from H5
    embs = []
    with h5py.File(pred_file, 'r') as f:
        embs.append(f['embeddings1'][:])
        if 'embeddings2' in f:
            embs.append(f['embeddings2'][:])

    assert args.clustering in ('ms', 'msplus', 'hdbscan', 'consistency'), \
        f"Unsupported clustering algorithm '{args.clustering}'. Supported values: ms, hdbscan, consistency"

    if args.ds_name == 'cityscapes':
        clustering = CityscapesClustering(args)
    else:
        clustering = CvpppClustering(args)

    return clustering(embs, pred_file)


def cluster_images(args):
    with futures.ProcessPoolExecutor(args.num_workers) as executor:
        pred_files = list(glob.glob(os.path.join(args.emb_dir, '*predictions.h5')))
        tasks = []
        # load h5 predictions files
        for pred_file in pred_files:
            filename = os.path.basename(pred_file)
            print(f'Processing {filename}')
            task = executor.submit(cluster_image, pred_file, args)
            tasks.append(task)

        results = [t.result() for t in tasks]

    print()
    for pf, r in zip(pred_files, results):
        print(f'{pf}: {r}')
    print()
    results = list(filter(lambda x: x is not None, results))
    results = np.array(results)
    print(f'Avg Segmentation Score: {np.mean(results)}')


if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.isdir(args.emb_dir)
    print(f"Clustering embeddings from '{args.emb_dir}' using '{args.clustering}'")
    cluster_images(args)
