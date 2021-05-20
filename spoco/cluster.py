import hdbscan
import numpy as np
import vigra
from numpy import linalg as LA
from scipy.ndimage import binary_erosion
from sklearn.cluster import DBSCAN, MeanShift


def iou(gt, seg):
    epsilon = 1e-5
    inter = (gt & seg).sum()
    union = (gt | seg).sum()

    iou = (inter + epsilon) / (union + epsilon)
    return iou


def expand_labels_watershed(seg, raw, erosion_iters=4):
    bg_mask = seg == 0
    # don't need to  do anything if we only have background
    if bg_mask.size == int(bg_mask.sum()):
        return seg

    hmap = vigra.filters.gaussianSmoothing(raw, sigma=1.)

    bg_mask = binary_erosion(bg_mask, iterations=erosion_iters)
    seg_new = seg.copy()
    bg_id = int(seg.max()) + 1
    seg_new[bg_mask] = bg_id

    seg_new, _ = vigra.analysis.watershedsNew(hmap, seeds=seg_new.astype('uint32'))

    seg_new[seg_new == bg_id] = 0
    return seg_new


def cluster(emb, clustering_alg, semantic_mask=None):
    output_shape = emb.shape[1:]
    # reshape (E, D, H, W) -> (E, D * H * W) and transpose -> (D * H * W, E)
    flattened_embeddings = emb.reshape(emb.shape[0], -1).transpose()

    result = np.zeros(flattened_embeddings.shape[0])

    if semantic_mask is not None:
        flattened_mask = semantic_mask.reshape(-1)
        assert flattened_mask.shape[0] == flattened_embeddings.shape[0]
    else:
        flattened_mask = np.ones(flattened_embeddings.shape[0])

    if flattened_mask.sum() == 0:
        # return zeros for empty masks
        return result.reshape(output_shape)

    # cluster only within the foreground mask
    clusters = clustering_alg.fit_predict(flattened_embeddings[flattened_mask == 1])
    # always increase the labels by 1 cause clustering results start from 0 and we may loose one object
    result[flattened_mask == 1] = clusters + 1

    return result.reshape(output_shape)


def cluster_hdbscan(emb, min_size, eps, min_samples=None, semantic_mask=None):
    clustering = hdbscan.HDBSCAN(min_cluster_size=min_size, cluster_selection_epsilon=eps, min_samples=min_samples)
    return cluster(emb, clustering, semantic_mask)


def cluster_dbscan(emb, eps, min_samples, semantic_mask=None):
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    return cluster(emb, clustering, semantic_mask)


def cluster_ms(emb, bandwidth, semantic_mask=None):
    clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    return cluster(emb, clustering, semantic_mask)


def cluster_consistency(emb1, emb2, eps, iou_threshold, num_anchors=100):
    """
    Consistency clustering as described in https://arxiv.org/abs/2103.14572
    """
    clustering = MeanShift(bandwidth=eps, bin_seeding=True)
    clusters = cluster(emb1, clustering)

    for l in np.unique(clusters):
        if l == 0:
            continue

        mask = clusters == l

        iou_table = []
        y, x = np.nonzero(mask)
        for _ in range(num_anchors):
            ind = np.random.randint(len(y))
            anchor_emb = emb2[:, y[ind], x[ind]]
            anchor_emb = anchor_emb[:, None, None]
            # compute the instance mask from emb2
            inst_mask = LA.norm(emb2 - anchor_emb, axis=0) < eps
            iou_table.append(iou(mask, inst_mask))

        median_iou = np.median(iou_table)

        if median_iou < iou_threshold:
            clusters[mask] = 0

    return clusters
