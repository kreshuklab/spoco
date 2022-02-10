import numpy as np
import torch
from skimage.metrics import contingency_table

from spoco.losses import compute_per_channel_dice, expand_as_one_hot


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and theTn simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        assert input.size() == target.size()

        per_batch_iou = []
        for _input, _target in zip(input, target):
            binary_prediction = self._binarize_predictions(_input, n_classes)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))

    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


def precision(tp, fp, fn):
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fp, fn):
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


def _relabel(input):
    _, unique_labels = np.unique(input, return_inverse=True)
    return unique_labels.reshape(input.shape)


def _iou_matrix(gt, seg):
    # relabel gt and seg for smaller memory footprint of contingency table
    gt = _relabel(gt)
    seg = _relabel(seg)

    # get number of overlapping pixels between GT and SEG
    n_inter = contingency_table(gt, seg).A

    # number of pixels for GT instances
    n_gt = n_inter.sum(axis=1, keepdims=True)
    # number of pixels for SEG instances
    n_seg = n_inter.sum(axis=0, keepdims=True)

    # number of pixels in the union between GT and SEG instances
    n_union = n_gt + n_seg - n_inter

    iou_matrix = n_inter / n_union
    # make sure that the values are within [0,1] range
    assert 0 <= np.min(iou_matrix) <= np.max(iou_matrix) <= 1

    return iou_matrix


class SegmentationMetrics:
    """
    Computes precision, recall, accuracy, f1 score for a given ground truth and predicted segmentation.
    Contingency table for a given ground truth and predicted segmentation is computed eagerly upon construction
    of the instance of `SegmentationMetrics`.

    Args:
        gt (ndarray): ground truth segmentation
        seg (ndarray): predicted segmentation
    """

    def __init__(self, gt, seg):
        self.iou_matrix = _iou_matrix(gt, seg)

    def metrics(self, iou_threshold):
        """
        Computes precision, recall, accuracy, f1 score at a given IoU threshold
        """
        # ignore background
        iou_matrix = self.iou_matrix[1:, 1:]
        detection_matrix = (iou_matrix > iou_threshold).astype(np.uint8)
        n_gt, n_seg = detection_matrix.shape

        # if the iou_matrix is empty or all values are 0
        trivial = min(n_gt, n_seg) == 0 or np.all(detection_matrix == 0)
        if trivial:
            tp = fp = fn = 0
        else:
            # count non-zero rows to get the number of TP
            tp = np.count_nonzero(detection_matrix.sum(axis=1))
            # count zero rows to get the number of FN
            fn = n_gt - tp
            # count zero columns to get the number of FP
            fp = n_seg - np.count_nonzero(detection_matrix.sum(axis=0))

        return {
            'precision': precision(tp, fp, fn),
            'recall': recall(tp, fp, fn),
            'accuracy': accuracy(tp, fp, fn),
            'f1': f1(tp, fp, fn)
        }


class Accuracy:
    """
    Computes accuracy between ground truth and predicted segmentation a a given threshold value.
    Defined as: AC = TP / (TP + FP + FN).
    Kaggle DSB2018 calls it Precision, see:
    https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric.
    """

    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold

    def __call__(self, input_seg, gt_seg):
        metrics = SegmentationMetrics(gt_seg, input_seg).metrics(self.iou_threshold)
        return metrics['accuracy']


class AveragePrecision:
    """
    Average precision taken for the IoU range (0.5, 0.95) with a step of 0.05 as defined in:
    https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric
    """

    def __init__(self, iou=None):
        if iou is not None:
            self.iou_range = [iou]
        else:
            self.iou_range = np.linspace(0.50, 0.95, 10)

    def __call__(self, input_seg, gt_seg):
        if len(np.unique(gt_seg)) == 1:
            return 1.

        # compute contingency_table
        sm = SegmentationMetrics(gt_seg, input_seg)
        # compute accuracy for each threshold
        acc = [sm.metrics(iou)['accuracy'] for iou in self.iou_range]
        # return the average
        return np.mean(acc)


def dice_score(gt, seg, smooth=1.):
    gt = gt > 0
    seg = seg > 0
    nom = 2 * np.sum(gt * seg)
    denom = np.sum(gt) + np.sum(seg)

    dice = float(nom + smooth) / float(denom + smooth)
    return dice


class DiceScore:
    def __call__(self, gt, seg, smooth=1.):
        return dice_score(gt, seg)


def best_dice(gt, seg):
    gt_lables = np.setdiff1d(np.unique(gt), [0])
    seg_labels = np.setdiff1d(np.unique(seg), [0])

    best_dices = []
    for gt_idx in gt_lables:
        _gt_seg = (gt == gt_idx).astype('uint8')
        dices = []
        for pred_idx in seg_labels:
            _pred_seg = (seg == pred_idx).astype('uint8')

            dice = dice_score(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    return np.mean(best_dices)


def symmetric_best_dice(gt, seg):
    bd1 = best_dice(gt, seg)
    bd2 = best_dice(seg, gt)
    return min(bd1, bd2)
