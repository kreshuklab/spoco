import math

import numpy as np
import torch
from torch import nn as nn


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class

    Returns:
        Cx1 tensor, i.e. Dice score for each channel
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


# embedding loss utils
def shift_tensor(tensor, offset):
    """ Shift a tensor by the given (spatial) offset.
    Arguments:
        tensor [torch.Tensor] - 4D (=2 spatial dims) or 5D (=3 spatial dims) tensor.
            Needs to be of float type.
        offset (tuple) - 2d or 3d spatial offset used for shifting the tensor
    """

    ndim = len(offset)
    assert ndim in (2, 3)
    diff = tensor.dim() - ndim

    # don't pad for the first dimensions
    # (usually batch and/or channel dimension)
    slice_ = diff * [slice(None)]

    # torch padding behaviour is a bit weird.
    # we use nn.ReplicationPadND
    # (torch.nn.functional.pad is even weirder and ReflectionPad is not supported in 3d)
    # still, padding needs to be given in the inverse spatial order

    # add padding in inverse spatial order
    padding = []
    for off in offset[::-1]:
        # if we have a negative offset, we need to shift "to the left",
        # which means padding at the right border
        # if we have a positive offset, we need to shift "to the right",
        # which means padding to the left border
        padding.extend([max(0, off), max(0, -off)])

    # add slicing in the normal spatial order
    for off in offset:
        if off == 0:
            slice_.append(slice(None))
        elif off > 0:
            slice_.append(slice(None, -off))
        else:
            slice_.append(slice(-off, None))

    # pad the spatial part of the tensor with replication padding
    slice_ = tuple(slice_)
    padding = tuple(padding)
    padder = nn.ReplicationPad2d if ndim == 2 else nn.ReplicationPad3d
    padder = padder(padding)
    shifted = padder(tensor)

    # slice the oadded tensor to get the spatially shifted tensor
    shifted = shifted[slice_]
    assert shifted.shape == tensor.shape

    return shifted


def invert_offsets(offsets):
    return [[-off for off in offset] for offset in offsets]


def embeddings_to_affinities(embeddings, offsets, delta):
    """ Transform embeddings to affinities.
    """
    # shift the embeddings by the offsets and stack them along a new axis
    # we need to shift in the opposite direction of the offsets, so we invert them
    # before applying the shift
    offsets_ = invert_offsets(offsets)
    shifted = torch.cat([shift_tensor(embeddings, off).unsqueeze(1) for off in offsets_], dim=1)
    # substract the embeddings from the shifted embeddings, take the norm and
    # transform to affinities based on the delta distance
    affs = (2 * delta - torch.norm(embeddings.unsqueeze(1) - shifted, dim=2)) / (2 * delta)
    affs = torch.clamp(affs, min=0) ** 2
    return affs


def segmentation_to_affinities(segmentation, offsets):
    """ Transform segmentation to affinities.
    Arguments:
        segmentation [torch.tensor] - 4D (2 spatial dims) or 5D (3 spatial dims) segmentation tensor.
            The channel axis (= dimension 1) needs to be a singleton.
        offsets [list[tuple]] - list of offsets for which to compute the affinities.
    """
    assert segmentation.shape[1] == 1
    # shift the segmentation and substract the shifted tensor from the segmentation
    # we need to shift in the opposite direction of the offsets, so we invert them
    # before applying the shift
    offsets_ = invert_offsets(offsets)
    shifted = torch.cat([shift_tensor(segmentation.float(), off) for off in offsets_], dim=1)
    affs = (segmentation - shifted)
    # the affinities are 1, where we had the same segment id (the difference is 0)
    # and 0 otherwise
    affs.eq_(0.)
    return affs


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


def flatten(tensor):
    """
    Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows: (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        output torch.Tensor of size (NxCxSPATIAL)
    """
    assert input.dim() > 2

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


################################################# embedding losses ####################################################

def compute_cluster_means(embeddings, target, n_instances):
    """
    Computes mean embeddings per instance, embeddings withing a given instance and the number of voxels per instance.

    C - number of instances
    E - embedding dimension
    SPATIAL - volume shape, i.e. DxHxW for 3D/ HxW for 2D

    Args:
        embeddings: tensor of pixel embeddings, shape: ExSPATIAL
        target: one-hot encoded target instances, shape: CxSPATIAL
    """
    target = expand_as_one_hot(target.unsqueeze(0), n_instances).squeeze(0)
    target = target.unsqueeze(1)
    spatial_ndim = embeddings.dim() - 1
    dim_arg = (2, 3) if spatial_ndim == 2 else (2, 3, 4)

    embedding_dim = embeddings.size(0)

    # get number of pixels in each cluster; output: Cx1
    num_pixels = torch.sum(target, dim=dim_arg)

    # expand target: Cx1xSPATIAL -> CxExSPATIAL
    shape = list(target.size())
    shape[1] = embedding_dim
    target = target.expand(shape)

    # expand input_: ExSPATIAL -> 1xExSPATIAL
    embeddings = embeddings.unsqueeze(0)

    # sum embeddings in each instance (multiply first via broadcasting); embeddings_per_instance shape CxExSPATIAL
    embeddings_per_instance = embeddings * target
    # num's shape: CxEx1(SPATIAL)
    num = torch.sum(embeddings_per_instance, dim=dim_arg)

    # compute mean embeddings per instance CxE
    mean_embeddings = num / num_pixels

    return mean_embeddings


class AbstractContrastiveLoss(nn.Module):
    """
    Implementation of contrastive loss defined in https://arxiv.org/pdf/1708.02551.pdf
    'Semantic Instance Segmentation with a Discriminative Loss Function'
    """

    def __init__(self, delta_var, delta_dist, norm='fro', alpha=1., beta=1., gamma=0.001, instance_term_weight=1.,
                 unlabeled_push_weight=1.,
                 ignore_label=None, bg_push=False, hinge_pull=True):
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.instance_term_weight = instance_term_weight
        self.unlabeled_push_weight = unlabeled_push_weight
        self.ignore_label = ignore_label
        self.bg_push = bg_push
        self.hinge_pull = hinge_pull

    def _compute_variance_term(self, cluster_means, embeddings, target, instance_counts, ignore_zero_label):
        """
        Computes the variance term, i.e. intra-cluster pull force that draws embeddings towards the mean embedding

        C - number of clusters (instances)
        E - embedding dimension
        SPATIAL - volume shape, i.e. DxHxW for 3D/ HxW for 2D

        Args:
            cluster_means: mean embedding of each instance, tensor (CxE)
            embeddings: embeddings vectors per instance, tensor (ExSPATIAL)
            target: label tensor (1xSPATIAL); each label is represented as one-hot vector
            instance_counts: number of voxels per instance
            ignore_zero_label: if True ignores the cluster corresponding to the 0-label
        """

        assert target.dim() in (2, 3)
        n_instances = cluster_means.shape[0]

        # compute the spatial mean and instance fields by scattering with the
        # target tensor
        cluster_means_spatial = cluster_means[target]
        instance_sizes_spatial = instance_counts[target]

        # permute the embedding dimension to axis 0
        if target.dim() == 2:
            cluster_means_spatial = cluster_means_spatial.permute(2, 0, 1)
        else:
            cluster_means_spatial = cluster_means_spatial.permute(3, 0, 1, 2)

        # compute the distance to cluster means
        dist_to_mean = torch.norm(embeddings - cluster_means_spatial, self.norm, dim=0)

        if ignore_zero_label:
            # zero out distances corresponding to 0-label cluster, so that it does not contribute to the loss
            dist_mask = torch.ones_like(dist_to_mean)
            dist_mask[target == 0] = 0
            dist_to_mean = dist_to_mean * dist_mask
            # decrease number of instances
            n_instances -= 1
            # if there is only 0-label in the target return 0
            if n_instances == 0:
                return 0.

        if self.hinge_pull:
            # zero out distances less than delta_var (hinge)
            dist_to_mean = torch.clamp(dist_to_mean - self.delta_var, min=0)

        dist_to_mean = dist_to_mean ** 2
        # normalize the variance by instance sizes and number of instances and sum it up
        variance_term = torch.sum(dist_to_mean / instance_sizes_spatial) / n_instances
        return variance_term

    def _compute_background_push(self, cluster_means, embeddings, target):
        assert target.dim() in (2, 3)
        n_instances = cluster_means.shape[0]

        # permute embedding dimension at the end
        if target.dim() == 2:
            embeddings = embeddings.permute(1, 2, 0)
        else:
            embeddings = embeddings.permute(1, 2, 3, 0)

        # decrease number of instances `C` since we're ignoring 0-label
        n_instances -= 1
        # if there is only 0-label in the target return 0
        if n_instances == 0:
            return 0.

        background_mask = target == 0
        n_background = background_mask.sum()
        background_push = 0.
        # skip embedding corresponding to the background pixels
        for cluster_mean in cluster_means[1:]:
            # compute distances between embeddings and a given cluster_mean
            dist_to_mean = torch.norm(embeddings - cluster_mean, self.norm, dim=-1)
            # apply background mask and compute hinge
            dist_hinged = torch.clamp((self.delta_dist - dist_to_mean) * background_mask, min=0) ** 2
            background_push += torch.sum(dist_hinged) / n_background

        # normalize by the number of instances
        return background_push / n_instances

    def _compute_distance_term(self, cluster_means, ignore_zero_label):
        """
        Compute the distance term, i.e an inter-cluster push-force that pushes clusters away from each other, increasing
        the distance between cluster centers

        Args:
            cluster_means: mean embedding of each instance, tensor (CxE)
            ignore_zero_label: if True ignores the cluster corresponding to the 0-label
        """
        C = cluster_means.size(0)
        if C == 1:
            # just one cluster in the batch, so distance term does not contribute to the loss
            return 0.

        # expand cluster_means tensor in order to compute the pair-wise distance between cluster means
        # CxE -> CxCxE
        cluster_means = cluster_means.unsqueeze(0)
        shape = list(cluster_means.size())
        shape[0] = C

        # cm_matrix1 is CxCxE
        cm_matrix1 = cluster_means.expand(shape)
        # transpose the cluster_means matrix in order to compute pair-wise distances
        cm_matrix2 = cm_matrix1.permute(1, 0, 2)
        # compute pair-wise distances between cluster means, result is a CxC tensor
        dist_matrix = torch.norm(cm_matrix1 - cm_matrix2, p=self.norm, dim=2)

        # create matrix for the repulsion distance (i.e. cluster centers further apart than 2 * delta_dist
        # are not longer repulsed)
        repulsion_dist = 2 * self.delta_dist * (1 - torch.eye(C))
        repulsion_dist = repulsion_dist.to(cluster_means.device)

        if ignore_zero_label:
            if C == 2:
                # just two cluster instances, including one which is ignored, i.e. distance term does not contribute to the loss
                return 0.
            # set the distance to 0-label to be greater than 2*delta_dist, so that it does not contribute to the loss because of the hinge at 2*delta_dist

            # find minimum dist
            d_min = torch.min(dist_matrix[dist_matrix > 0]).item()
            # dist_multiplier = 2 * delta_dist / d_min + unlabeled_push_weight
            dist_multiplier = 2 * self.delta_dist / d_min + 1e-3
            # create distance mask
            dist_mask = torch.ones_like(dist_matrix)
            dist_mask[0, 1:] = dist_multiplier
            dist_mask[1:, 0] = dist_multiplier

            # mask the dist_matrix
            dist_matrix = dist_matrix * dist_mask
            # decrease number of instances
            C -= 1

        # zero out distances grater than 2*delta_dist (hinge)
        hinged_dist = torch.clamp(repulsion_dist - dist_matrix, min=0) ** 2
        # sum all of the hinged pair-wise distances
        dist_sum = torch.sum(hinged_dist)
        # normalized by the number of paris and return
        distance_term = dist_sum / (C * (C - 1))
        return distance_term

    def _compute_regularizer_term(self, cluster_means):
        """
        Computes the regularizer term, i.e. a small pull-force that draws all clusters towards origin to keep
        the network activations bounded
        """
        # compute the norm of the mean embeddings
        norms = torch.norm(cluster_means, p=self.norm, dim=1)
        # return the average norm per batch
        return torch.sum(norms) / cluster_means.size(0)

    def instance_based_loss(self, embeddings, cluster_means, target):
        """
        Computes auxiliary loss based on embeddings and a given list of target instances together with their mean embeddings

        Args:
            embeddings (torch.tensor): pixel embeddings (ExSPATIAL)
            cluster_means (torch.tensor): mean embeddings per instance (CxExSINGLETON_SPATIAL)
            target (torch.tensor): ground truth instance segmentation (SPATIAL)
        """
        raise NotImplementedError

    def forward(self, input_, target):
        """
        Args:
             input_ (torch.tensor): embeddings predicted by the network (NxExDxHxW) (E - embedding dims)
                                    expects float32 tensor
             target (torch.tensor): ground truth instance segmentation (NxDxHxW)
                                    expects int64 tensor
                                    if self.ignore_zero_label is True then expects target of shape Nx2xDxHxW where
                                    relabeled version is in target[:,0,...] and the original labeling is in target[:,1,...]

        Returns:
            Combined loss defined as: alpha * variance_term + beta * distance_term + gamma * regularization_term
        """

        n_batches = input_.shape[0]
        # compute the loss per each instance in the batch separately
        # and sum it up in the per_instance variable
        per_instance_loss = 0.
        for single_input, single_target in zip(input_, target):
            # check if the target contain ignore_label; ignore_label is going to be mapped to the 0-label
            # so we just need to ignore 0-label in the pull and push forces
            ignore_zero_label, single_target = self._should_ignore(single_target)
            contains_bg = 0 in single_target
            if self.bg_push and contains_bg:
                ignore_zero_label = True

            instance_ids, instance_counts = torch.unique(single_target, return_counts=True)

            # compare spatial dimensions
            assert single_input.size()[1:] == single_target.size()

            # compute mean embeddings (output is of shape CxE)
            cluster_means = compute_cluster_means(single_input, single_target, instance_ids.size(0))

            # compute variance term, i.e. pull force
            variance_term = self._compute_variance_term(cluster_means, single_input, single_target, instance_counts,
                                                        ignore_zero_label)

            # compute background push force, i.e. push force between the mean cluster embeddings and embeddings of background pixels
            # compute only ignore_zero_label is True, i.e. a given patch contains background label
            unlabeled_push = 0.
            if self.bg_push and contains_bg:
                unlabeled_push = self._compute_background_push(cluster_means, single_input, single_target)

            # compute the instance-based loss
            instance_loss = self.instance_based_loss(single_input, cluster_means, single_target)

            # compute distance term, i.e. push force
            distance_term = self._compute_distance_term(cluster_means, ignore_zero_label)

            # compute regularization term
            regularization_term = self._compute_regularizer_term(cluster_means)

            # compute total loss and sum it up
            loss = self.alpha * variance_term + \
                   self.beta * distance_term + \
                   self.gamma * regularization_term + \
                   self.instance_term_weight * instance_loss + \
                   self.unlabeled_push_weight * unlabeled_push

            per_instance_loss += loss

        # reduce across the batch dimension
        return per_instance_loss.div(n_batches)

    def _should_ignore(self, target):
        # set default values
        ignore_zero_label = False
        single_target = target

        if self.ignore_label is not None:
            assert target.dim() == 4, "Expects target to be 2xDxHxW when ignore_label is set"
            # get relabeled target
            single_target = target[0]
            # get original target and ignore 0-label only if 0-label was present in the original target
            original = target[1]
            ignore_zero_label = self.ignore_label in original

        return ignore_zero_label, single_target


class ContrastiveLoss(AbstractContrastiveLoss):
    def __init__(self, delta_var, delta_dist, norm='fro', alpha=1., beta=1., gamma=0.001, ignore_label=None,
                 bg_push=False, hinge_pull=True, **kwargs):
        super(ContrastiveLoss, self).__init__(delta_var, delta_dist, norm=norm,
                                              alpha=alpha, beta=beta, gamma=gamma, instance_term_weight=0.,
                                              unlabeled_push_weight=0.,
                                              ignore_label=ignore_label, bg_push=bg_push, hinge_pull=hinge_pull)

    def instance_based_loss(self, embeddings, cluster_means, target):
        # no auxiliary loss in the standard ContrastiveLoss
        return 0.


class AffinitySideLoss(nn.Module):
    eps = 1.e-6

    def __init__(self, delta_dist, offset_ranges, n_samples):
        super().__init__()
        self.offset_ranges = offset_ranges
        self.n_samples = n_samples
        assert all(len(orange) == 2 for orange in self.offset_ranges)
        self.ndim = len(self.offset_ranges)
        self.delta = delta_dist
        self.dice_loss = DiceLoss(normalization='none')

    def __call__(self, input_, target):
        # add batch and channel dim
        target = target.unsqueeze(0).unsqueeze(0)
        input_ = input_.unsqueeze(0)
        assert input_.dim() == target.dim()
        assert input_.shape[2:] == target.shape[2:]

        # sample offsets
        offsets = [[np.random.randint(orange[0], orange[1]) for orange in self.offset_ranges]
                   for _ in range(self.n_samples)]

        # we invert the affinities and the target affinities
        # so that we get boundaries as foreground, which is beneficial for
        # the dice loss.
        # compute affinities from emebeddings
        affs = 1. - embeddings_to_affinities(input_, offsets, self.delta)

        # compute groundtruth affinities from target
        target_affs = 1. - segmentation_to_affinities(target, offsets)
        assert affs.shape == target_affs.shape, "%s, %s" % (str(affs.shape),
                                                            str(target_affs.shape))

        # compute the dice score between affinities and target affinities
        loss = self.dice_loss(affs, target_affs)
        return loss


class SpocoContrastiveLoss(AbstractContrastiveLoss):
    def __init__(self, delta_var, delta_dist, instance_loss, kernel_threshold,
                 norm='fro', alpha=1., beta=1., gamma=0.001, instance_term_weight=1., unlabeled_push_weight=1.,
                 ignore_label=None, bg_push=False, hinge_pull=True, aux_loss_ignore_zero=True, **kwargs):

        super().__init__(delta_var, delta_dist, norm=norm,
                         alpha=alpha, beta=beta, gamma=gamma, instance_term_weight=instance_term_weight,
                         unlabeled_push_weight=unlabeled_push_weight,
                         ignore_label=ignore_label, bg_push=bg_push, hinge_pull=hinge_pull)

        self.aux_loss_ignore_zero = aux_loss_ignore_zero
        # ignore instance corresponding to 0-label
        self.delta_var = delta_var
        # init auxiliary loss
        # TODO: add BCE loss
        assert instance_loss in ['dice', 'affinity', 'dice_aff']
        if instance_loss == 'dice':
            self.instance_loss = DiceLoss(normalization='none')
        elif instance_loss == 'affinity':
            self.instance_loss = AffinitySideLoss(
                delta_dist=delta_dist,
                offset_ranges=kwargs.get('offset_ranges', [(-18, 0), (-18, 0)]),
                n_samples=kwargs.get('n_samples', 9)
            )
        elif instance_loss == 'dice_aff':
            # combine dice and affinity side loss
            dice_weight = kwargs.get('dice_weight', 1.0)
            aff_weight = kwargs.get('aff_weight', 1.0)

            dice_loss = DiceLoss(normalization='none')
            aff_loss = AffinitySideLoss(
                delta_dist=delta_dist,
                offset_ranges=kwargs.get('offset_ranges', [(-18, 0), (-18, 0)]),
                n_samples=kwargs.get('n_samples', 9)
            )

            self.instance_loss = CombinedAuxLoss(
                losses=[dice_loss, aff_loss],
                weights=[dice_weight, aff_weight]
            )

        # init dist_to_mask function which maps per-instance distance map to the instance probability map
        self.dist_to_mask = self.Gaussian(delta_var=delta_var, pmaps_threshold=kernel_threshold)

    def create_instance_pmaps_and_masks(self, embeddings, anchors, target):
        """
        Given the feature space and the anchor embeddings returns the 'soft' masks (one for every anchor)
        together with ground truth binary masks extracted from the target.

        Both: 'soft' masks and ground truth masks are stacked along a new channel dimension.

        Args:
            embeddings (torch.Tensor): ExSpatial image embeddings (E - emb dim)
            anchors (torch.Tensor): CxE anchor points in the embedding space (C - number of anchors)
            target (torch.Tensor): (partial) ground truth segmentation

        Returns:
            (soft_masks, gt_masks): tuple of two tensors of shape CxSpatial
        """
        inst_pmaps = []
        inst_masks = []

        # permute embedding dimension
        if target.dim() == 2:
            embeddings = embeddings.permute(1, 2, 0)
        else:
            embeddings = embeddings.permute(1, 2, 3, 0)

        for i, anchor_emb in enumerate(anchors):
            if i == 0 and self.aux_loss_ignore_zero:
                # ignore 0-label
                continue
            # compute distance map
            distance_map = torch.norm(embeddings - anchor_emb, self.norm, dim=-1)
            # convert distance map to instance pmaps and save
            inst_pmaps.append(self.dist_to_mask(distance_map).unsqueeze(0))
            # create real mask and save
            assert i in target
            inst_masks.append((target == i).float().unsqueeze(0))

        if not inst_masks:
            # no masks have been extracted from the image
            return None, None

        # stack along batch dimension
        inst_pmaps = torch.stack(inst_pmaps)
        inst_masks = torch.stack(inst_masks)

        return inst_pmaps, inst_masks

    def instance_based_loss(self, embeddings, cluster_means, target):
        assert embeddings.size()[1:] == target.size()

        if isinstance(self.instance_loss, AffinitySideLoss):
            # compute just the affinity side loss
            return self.instance_loss(embeddings, target)
        else:
            # extract soft and ground truth masks from the feature space
            instance_pmaps, instance_masks = self.create_instance_pmaps_and_masks(embeddings, cluster_means, target)

            if isinstance(self.instance_loss, CombinedAuxLoss):
                # compute combined affinity and instance-based loss
                return self.instance_loss(embeddings, target, instance_pmaps, instance_masks)
            else:
                # compute instance-based loss
                if instance_masks is None:
                    return 0.
                return self.instance_loss(instance_pmaps, instance_masks).mean()

    # kernel function used to convert the distance map (i.e. `||embeddings - anchor_embedding||`) into an instance mask
    class Gaussian(nn.Module):
        def __init__(self, delta_var, pmaps_threshold):
            super().__init__()
            self.delta_var = delta_var
            # dist_var^2 = -2*sigma*ln(pmaps_threshold)
            self.two_sigma = delta_var * delta_var / (-math.log(pmaps_threshold))

        def forward(self, dist_map):
            return torch.exp(- dist_map * dist_map / self.two_sigma)


class SpocoConsistencyContrastiveLoss(SpocoContrastiveLoss):
    def __init__(self, delta_var, delta_dist, instance_loss, kernel_threshold, norm='fro', alpha=1., beta=1.,
                 gamma=0.001, instance_term_weight=1., unlabeled_push_weight=1., ignore_label=None, bg_push=True,
                 hinge_pull=True, aux_loss_ignore_zero=True, joint_loss=False, consistency_weight=1.0, max_anchors=20,
                 volume_threshold=0.05, consistency_only=False, **kwargs):

        super().__init__(delta_var, delta_dist, instance_loss, kernel_threshold, norm, alpha, beta, gamma,
                         instance_term_weight,
                         unlabeled_push_weight,
                         ignore_label, bg_push, hinge_pull, aux_loss_ignore_zero, **kwargs)

        self.consistency_weight = consistency_weight
        self.max_anchors = max_anchors
        self.volume_threshold = volume_threshold
        self.dice_loss = DiceLoss(normalization='none')
        self.joint_loss = joint_loss
        self.consistency_only = consistency_only

    def _inst_pmap(self, emb, anchor, mask):
        # compute distance map
        distance_map = torch.norm(emb - anchor, self.norm, dim=0)
        # compute hard mask, i.e. delta_var-neighborhood and zero-out in the mask
        inst_mask = distance_map < self.delta_var
        mask[inst_mask] = 0
        # convert distance map to instance pmaps and return
        return self.dist_to_mask(distance_map)

    def emb_consistency(self, emb_f, emb_g, mask):
        soft_masks_f = []
        soft_masks_g = []
        for i in range(self.max_anchors):
            if mask.sum() < self.volume_threshold * mask.numel():
                break

            try:
                # get random anchor
                indices = torch.nonzero(mask, as_tuple=True)
                ind = np.random.randint(len(indices[0]))

                f_mask = self._extract_pmap(emb_f, mask, indices, ind)
                g_mask = self._extract_pmap(emb_g, mask, indices, ind)
            except IndexError as e:
                # non-deterministic error workaround
                print(f'ERROR: {e}')
                continue

            soft_masks_f.append(f_mask)
            soft_masks_g.append(g_mask)

        # stack along channel dim
        soft_masks_f = torch.stack(soft_masks_f)
        soft_masks_g = torch.stack(soft_masks_g)

        return self.dice_loss(soft_masks_f, soft_masks_g)

    def _extract_pmap(self, emb, mask, indices, ind):
        if mask.dim() == 2:
            y, x = indices
            anchor = emb[:, y[ind], x[ind]]
            anchor = anchor[:, None, None]
        else:
            z, y, x = indices
            anchor = emb[:, z[ind], y[ind], x[ind]]
            anchor = anchor[:, None, None, None]

        return self._inst_pmap(emb, anchor, mask)

    def forward(self, _input, target):
        assert len(_input) == 2
        emb_f, emb_g = _input

        if not self.consistency_only:
            # compute contrastive loss on the embeddings coming from q
            contrastive_loss = super().forward(emb_f, target)
        else:
            contrastive_loss = 0

        if self.joint_loss:
            # compute contrastive loss on the embeddings coming from g
            loss_g = super().forward(emb_g, target)
            contrastive_loss = (contrastive_loss + loss_g) / 2

        for e_f, e_g, t in zip(emb_f, emb_g, target):
            unlabeled_mask = (t == 0).int()
            if unlabeled_mask.sum() < self.volume_threshold * unlabeled_mask.numel():
                continue
            emb_consistency_loss = self.emb_consistency(e_f, e_g, unlabeled_mask)
            contrastive_loss += self.consistency_weight * emb_consistency_loss

        return contrastive_loss


class CombinedAuxLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, embeddings, target, instance_pmaps, instance_masks):
        result = 0.
        for loss, weight in zip(self.losses, self.weights):
            if isinstance(loss, AffinitySideLoss):
                result += weight * loss(embeddings, target)
            else:
                if instance_masks is not None:
                    result += weight * loss(instance_pmaps, instance_masks).mean()
        return result


def create_loss(delta_var, delta_dist, alpha, beta, gamma, unlabeled_push_weight, instance_term_weight,
                consistency_weight, kernel_threshold, instance_loss, spoco):
    """
    Creates an instance of the embedding loss based on the parameters provided.
    If `unlabeled_push_weight` is set to zero ContrastiveLos or SpocoContrastiveLoss (if `instance_term_weight`
    is greater than 0) is returned, otherwise the SpocoConsistencyContrastiveLoss (sparse setting) is returned.

    Args:
        spoco: use spoco setting
        delta_var: pull force hinge
        delta_dist: push force hinge
        alpha: pull force weight
        beta: push force weight
        gamma: regularizer term weight
        unlabeled_push_weight: unlabeled push force weight
        instance_term_weight: instance-based loss weight
        consistency_weight: embedding consistency loss weight
        kernel_threshold: threshold used for the differentiable instance selection
        instance_loss: name of the instance-based loss ('dice', 'affinity', 'dice_aff', 'bce')

    Returns:
        nn.Module: an instance of the loss function

    """
    assert delta_var > 0
    assert delta_dist > 0

    if spoco:
        bg_push = unlabeled_push_weight > 0
        print('Unlabeled push: ', bg_push)
        return SpocoConsistencyContrastiveLoss(delta_var=delta_var, delta_dist=delta_dist,
                                               alpha=alpha, beta=beta, gamma=gamma,
                                               instance_loss=instance_loss, kernel_threshold=kernel_threshold,
                                               unlabeled_push_weight=unlabeled_push_weight,
                                               instance_term_weight=instance_term_weight,
                                               consistency_term_weight=consistency_weight,
                                               bg_push=bg_push)
    else:
        if instance_term_weight > 0:
            return SpocoContrastiveLoss(delta_var=delta_var, delta_dist=delta_dist,
                                        alpha=alpha, beta=beta, gamma=gamma,
                                        unlabeled_push_weight=0.,
                                        instance_term_weight=instance_term_weight,
                                        instance_loss=instance_loss, kernel_threshold=kernel_threshold)
        else:
            return ContrastiveLoss(delta_var=delta_var, delta_dist=delta_dist,
                                   alpha=alpha, beta=beta, gamma=gamma)
