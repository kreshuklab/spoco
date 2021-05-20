import numpy as np
import torch
from numpy import linalg as LA
from skimage import measure
from skimage.metrics import adapted_rand_error, contingency_table

from spoco.losses import compute_per_channel_dice, expand_as_one_hot
from spoco.utils import convert_to_numpy


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


class AdaptedRandError:
    """
    A functor which computes an Adapted Rand error as defined by the SNEMI3D contest
    (http://brainiac2.mit.edu/SNEMI3D/evaluation).

    This is a generic implementation which takes the input, converts it to the segmentation image (see `input_to_segm()`)
    and then computes the ARand between the segmentation and the ground truth target. Depending on one's use case
    it's enough to extend this class and implement the `input_to_segm` method.

    Args:
        use_last_target (bool): use only the last channel from the target to compute the ARand
    """

    def __init__(self, use_last_target=False, **kwargs):
        self.use_last_target = use_last_target

    def __call__(self, input, target):
        """
        Compute ARand Error for each input, target pair in the batch and return the mean value.

        Args:
            input (torch.tensor): 5D (NCDHW) output from the network
            target (torch.tensor): 4D (NDHW) ground truth segmentation

        Returns:
            average ARand Error across the batch
        """

        def _arand_err(gt, seg):
            n_seg = len(np.unique(seg))
            if n_seg == 1:
                return 0.
            return adapted_rand_error(gt, seg)[0]

        # converts input and target to numpy arrays
        input, target = convert_to_numpy(input, target)
        if self.use_last_target:
            target = target[:, -1, ...]  # 4D
        else:
            # use 1st target channel
            target = target[:, 0, ...]  # 4D

        # ensure target is of integer type
        target = target.astype(np.int)

        per_batch_arand = []
        for _input, _target in zip(input, target):
            n_clusters = len(np.unique(_target))
            # skip ARand eval if there is only one label in the patch due to the zero-division error in Arand impl
            # xxx/skimage/metrics/_adapted_rand_error.py:70: RuntimeWarning: invalid value encountered in double_scalars
            # precision = sum_p_ij2 / sum_a2
            if n_clusters == 1:
                per_batch_arand.append(0.)
                continue

            # convert _input to segmentation CDHW
            segm = self.input_to_segm(_input)
            assert segm.ndim == 4

            # compute per channel arand and return the minimum value
            per_channel_arand = [_arand_err(_target, channel_segm) for channel_segm in segm]
            per_batch_arand.append(np.min(per_channel_arand))

        # return mean arand error
        mean_arand = torch.mean(torch.tensor(per_batch_arand))
        return mean_arand

    def input_to_segm(self, input):
        """
        Converts input tensor (output from the network) to the segmentation image. E.g. if the input is the boundary
        pmaps then one option would be to threshold it and run connected components in order to return the segmentation.

        :param input: 4D tensor (CDHW)
        :return: segmentation volume either 4D (segmentation per channel)
        """
        # by deafult assume that input is a segmentation volume itself
        return input


class GenericAdaptedRandError(AdaptedRandError):
    def __init__(self, input_channels, thresholds=None, use_last_target=True, invert_channels=None, **kwargs):

        super().__init__(use_last_target=use_last_target, **kwargs)
        assert isinstance(input_channels, list) or isinstance(input_channels, tuple)
        self.input_channels = input_channels
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        if invert_channels is None:
            invert_channels = []
        self.invert_channels = invert_channels

    def input_to_segm(self, input):
        # pick only the channels specified in the input_channels
        results = []
        for i in self.input_channels:
            c = input[i]
            # invert channel if necessary
            if i in self.invert_channels:
                c = 1 - c
            results.append(c)

        input = np.stack(results)

        segs = []
        for predictions in input:
            for th in self.thresholds:
                # run connected components on the predicted mask; consider only 1-connectivity
                seg = measure.label((predictions > th).astype(np.uint8), background=0, connectivity=1)
                segs.append(seg)

        return np.stack(segs)


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

    def __init__(self):
        self.iou_range = np.linspace(0.50, 0.95, 10)

    def __call__(self, input_seg, gt_seg):
        # compute contingency_table
        sm = SegmentationMetrics(gt_seg, input_seg)
        # compute accuracy for each threshold
        acc = [sm.metrics(iou)['accuracy'] for iou in self.iou_range]
        # return the average
        return np.mean(acc)


class GenericAveragePrecision:
    def __init__(self, min_instance_size=None, use_last_target=False, metric='ap', **kwargs):
        self.min_instance_size = min_instance_size
        self.use_last_target = use_last_target
        assert metric in ['ap', 'acc']
        if metric == 'ap':
            # use AveragePrecision
            self.metric = AveragePrecision()
        else:
            # use Accuracy at 0.5 IoU
            self.metric = Accuracy(iou_threshold=0.5)

    def __call__(self, input, target):
        if self.use_last_target:
            target = target[:, -1, ...]  # 4D

        input, target = convert_to_numpy(input, target)

        batch_aps = []
        i_batch = 0
        # iterate over the batch
        for inp, tar in zip(input, target):
            segs = self.input_to_seg(inp, tar)

            # convert target to seg
            tar = self.target_to_seg(tar)

            # filter small instances if necessary
            tar = self._filter_instances(tar)

            # compute average precision per channel
            segs_aps = [self.metric(self._filter_instances(seg), tar) for seg in segs]

            # save max AP
            batch_aps.append(np.max(segs_aps))
            i_batch += 1

        return torch.tensor(batch_aps).mean()

    def _filter_instances(self, input):
        """
        Filters instances smaller than 'min_instance_size' by overriding them with 0-index
        :param input: input instance segmentation
        """
        if self.min_instance_size is not None:
            labels, counts = np.unique(input, return_counts=True)
            for label, count in zip(labels, counts):
                if count < self.min_instance_size:
                    input[input == label] = 0
        return input

    def input_to_seg(self, input, target=None):
        raise NotImplementedError

    def target_to_seg(self, target):
        return target


class BlobsAveragePrecision(GenericAveragePrecision):
    """
    Computes Average Precision given foreground prediction and ground truth instance segmentation.
    """

    def __init__(self, thresholds=None, metric='ap', min_instance_size=None, input_channel=0, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True, metric=metric)
        if thresholds is None:
            thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.input_channel = input_channel

    def input_to_seg(self, input, target=None):
        input = input[self.input_channel]
        segs = []
        for th in self.thresholds:
            # threshold and run connected components
            mask = (input > th).astype(np.uint8)
            seg = measure.label(mask, background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)


class MeanEmbeddingAveragePrecision(GenericAveragePrecision):
    """
    Computes the AP based on pixel embeddings the ground truth instance segmentation.
    The following algorithm is used to get the instance segmentation
        for i in ground_truth_instances:
            1. get average embedding in instance i
            2. get the object mask by growing the epsilon ball around the pixel's embedding
            3. add the object to the list of instances
    """

    def __init__(self, epsilon, min_instance_size=None, metric='ap', **kwargs):
        super().__init__(min_instance_size, use_last_target=False, metric=metric, **kwargs)
        self.epsilon = epsilon

    def input_to_seg(self, embeddings, target=None):
        assert target is not None

        result = np.zeros(shape=embeddings.shape[1:], dtype=np.uint32)

        spatial_dims = (1, 2) if result.ndim == 2 else (1, 2, 3)

        labels, counts = np.unique(target, return_counts=True)
        for label, size in zip(labels, counts):
            # skip 0-label
            if label == 0:
                continue

            # get the mask for this instance
            instance_mask = (target == label)

            # mask out all embeddings not in this instance
            embeddings_per_instance = embeddings * instance_mask

            # compute the cluster mean
            mean_embedding = np.sum(embeddings_per_instance, axis=spatial_dims, keepdims=True) / size
            # compute the instance mask, i.e. get the epsilon-ball
            inst_mask = LA.norm(embeddings - mean_embedding, axis=0) < self.epsilon
            # save instance
            result[inst_mask] = label

        return np.expand_dims(result, 0)


class RandomEmbeddingAveragePrecision(GenericAveragePrecision):
    def __init__(self, epsilon, min_instance_size=None, metric='ap', **kwargs):
        super().__init__(min_instance_size, use_last_target=False, metric=metric, **kwargs)
        self.epsilon = epsilon

    def __str__(self):
        return f"RandomEmbeddingAveragePrecision(epsilon: {self.epsilon})"

    def input_to_seg(self, embeddings, target=None):
        assert target is not None

        result = np.zeros(shape=embeddings.shape[1:], dtype=np.uint32)

        labels = np.unique(target)

        for label in labels:
            # skip 0-label
            if label == 0:
                continue

            indices = np.nonzero(target == label)
            assert len(indices) in (2, 3)

            # pick random point
            ind = np.random.randint(len(indices[0]))

            if len(indices) == 2:
                y, x = indices
                anchor_emb = embeddings[:, y[ind], x[ind]]
                anchor_emb = anchor_emb[:, None, None]
            else:
                z, y, x = indices
                anchor_emb = embeddings[:, z[ind], y[ind], x[ind]]
                anchor_emb = anchor_emb[:, None, None, None]

            # compute the instance mask, i.e. get the epsilon-ball
            inst_mask = LA.norm(embeddings - anchor_emb, axis=0) < self.epsilon

            # save instance
            result[inst_mask] = label

        return np.expand_dims(result, 0)


class CVPPPEmbeddingDiceScore:
    def __init__(self, epsilon, max_anchors, refine_iters=5, **kwargs):
        self.max_anchors = max_anchors
        self.epsilon = epsilon
        self.refine_iters = refine_iters

    def __call__(self, input, target):
        def _dice_score(gt, seg):
            nom = 2 * np.sum(gt * seg)
            denom = np.sum(gt) + np.sum(seg)
            dice = float(nom) / float(denom)
            return dice

        # input NxExDxHxW, target NxDxHxW
        input = input.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        batch_dice = []
        # iterate over the batch
        for inp, tar in zip(input, target):
            seg = self.emb_to_seg(inp, tar)
            # convert target to binary mask
            mask_tar = (tar > 0).astype('uint8')
            mask_seg = (seg > 0).astype('uint8')
            # compute dice score
            ds = _dice_score(mask_tar, mask_seg)
            batch_dice.append(ds)

        return torch.tensor(batch_dice).mean()

    def __str__(self):
        return f"CVPPPEmbeddingDiceScore(epsilon: {self.epsilon}, max_anchors: {self.max_anchors})"

    @staticmethod
    def refine_instance(emb, mask, eps, iters):
        for _ in range(iters):
            num_pixels = np.sum(mask)
            mask_emb = emb * mask
            num = np.sum(mask_emb, axis=(1, 2), keepdims=True)

            mean_emb = num / num_pixels
            mask = LA.norm(emb - mean_emb, axis=0) < eps
        return mask

    def emb_to_seg(self, embeddings, target):
        assert embeddings.ndim == 3
        assert target.ndim == 2

        result = np.zeros(shape=embeddings.shape[1:], dtype=np.uint32)
        mask = target > 0

        # for sparse objects we might have empty patches, just return the target mask
        if np.sum(mask) == 0:
            return np.ones_like(mask)

        for i in range(self.max_anchors):
            if np.sum(mask) == 0:
                return result
            # get random anchor
            y, x = np.nonzero(mask)
            ind = np.random.randint(len(y))
            anchor_emb = embeddings[:, y[ind], x[ind]]
            anchor_emb = anchor_emb[:, None, None]
            # compute the instance mask, i.e. get the epsilon-ball
            inst_mask = LA.norm(embeddings - anchor_emb, axis=0) < self.epsilon
            # refine instance mask
            inst_mask = self.refine_instance(embeddings, inst_mask, self.epsilon, iters=self.refine_iters)
            # zero out the instance in the mask
            mask[inst_mask] = 0
            # save instance
            result[inst_mask] = i + 1

        return result


def create_eval_metric(ds_name, loss_delta_var):
    if ds_name == 'cvppp':
        return CVPPPEmbeddingDiceScore(epsilon=loss_delta_var, max_anchors=20)
    else:
        return RandomEmbeddingAveragePrecision(epsilon=loss_delta_var)
