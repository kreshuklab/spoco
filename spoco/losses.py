import numpy as np
import torch
from torch import nn as nn
from torch_scatter import scatter_mean

from spoco.utils import shift_tensor, GaussianKernel


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


def select_stable_anchor(embeddings, mean_embedding, object_mask, delta_var, norm='fro'):
    """
    Anchor sampling procedure. Given a binary mask of an object (`object_mask`) and a `mean_embedding` vector within
    the mask, the function selects a pixel from the mask at random and returns its embedding only if it's closer than
    `delta_var` from the `mean_embedding`.

    Args:
        embeddings (torch.Tensor): ExSpatial vector field of an image
        mean_embedding (torch.Tensor): E-dimensional mean of embeddings lying within the `object_mask`
        object_mask (torch.Tensor): binary image of a selected object
        delta_var (float): contrastive loss, pull force margin
        norm (str): vector norm used, default: Frobenius norm

    Returns:
        embedding of a selected pixel within the mask or the mean embedding if stable anchor could be found
    """
    indices = torch.nonzero(object_mask, as_tuple=True)
    # convert to numpy
    indices = [t.cpu().numpy() for t in indices]

    # randomize coordinates
    seed = np.random.randint(np.iinfo('int32').max)
    for t in indices:
        rs = np.random.RandomState(seed)
        rs.shuffle(t)

    for ind in range(len(indices[0])):
        if object_mask.dim() == 2:
            y, x = indices
            anchor_emb = embeddings[:, y[ind], x[ind]]
            anchor_emb = anchor_emb[..., None, None]
        else:
            z, y, x = indices
            anchor_emb = embeddings[:, z[ind], y[ind], x[ind]]
            anchor_emb = anchor_emb[..., None, None, None]
        dist_to_mean = torch.norm(mean_embedding - anchor_emb, norm)
        if dist_to_mean < delta_var:
            return anchor_emb
    # if stable anchor has not been found, return mean_embedding
    return mean_embedding


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
    Computes mean embeddings per instance.
    E - embedding dimension

    Args:
        embeddings: tensor of pixel embeddings, shape: ExSPATIAL
        target: one-hot encoded target instances, shape: SPATIAL
        n_instances: number of instances
    """
    embeddings = embeddings.flatten(1)
    target = target.flatten()
    mean_embeddings = scatter_mean(embeddings, target, dim_size=n_instances)
    return mean_embeddings.transpose(1, 0)


class AbstractContrastiveLoss(nn.Module):
    """
    Abstract class for all contrastive-based losses.
    This implementation expands all tensors to match the instance dimensions.
    This means that it's fast, but has high memory footprint.
    """

    def __init__(self, delta_var, delta_dist, norm='fro', alpha=1., beta=1., gamma=0.001, unlabeled_push_weight=0.0,
                 instance_term_weight=1.0):
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.unlabeled_push_weight = unlabeled_push_weight
        self.unlabeled_push = unlabeled_push_weight > 0
        self.instance_term_weight = instance_term_weight

    def __str__(self):
        return super().__str__() + f"\ndelta_var: {self.delta_var}\ndelta_dist: {self.delta_dist}" \
                                   f"\nalpha: {self.alpha}\nbeta: {self.beta}\ngamma: {self.gamma}" \
                                   f"\nunlabeled_push_weight: {self.unlabeled_push_weight}" \
                                   f"\ninstance_term_weight: {self.instance_term_weight}"

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

        # zero out distances less than delta_var (hinge)
        hinge_dist = torch.clamp(dist_to_mean - self.delta_var, min=0) ** 2

        # normalize the variance by instance sizes and number of instances and sum it up
        variance_term = torch.sum(hinge_dist / instance_sizes_spatial) / n_instances
        return variance_term

    def _compute_unlabeled_push(self, cluster_means, embeddings, target):
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
            # dist_multiplier = 2 * delta_dist / d_min + epsilon
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

    def compute_instance_term(self, embeddings, cluster_means, target):
        """
        Computes auxiliary loss based on embeddings and a given list of target instances together with their mean embeddings

        Args:
            embeddings (torch.tensor): pixel embeddings (ExSPATIAL)
            cluster_means (torch.tensor): mean embeddings per instance (CxExSINGLETON_SPATIAL)
            target (torch.tensor): ground truth instance segmentation (SPATIAL)

        Returns:
            float: value of the instance-based term
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
                                        + instance_term_weight * instance_term + unlabeled_push_weight * unlabeled_push_term
        """

        n_batches = input_.shape[0]
        # compute the loss per each instance in the batch separately
        # and sum it up in the per_instance variable
        loss = 0.
        for single_input, single_target in zip(input_, target):
            contains_bg = 0 in single_target
            if self.unlabeled_push and contains_bg:
                ignore_zero_label = True

            # get number of instances in the batch instance
            instance_ids, instance_counts = torch.unique(single_target, return_counts=True)

            # get the number of instances
            C = instance_ids.size(0)

            # compare spatial dimensions
            assert single_input.size()[1:] == single_target.size()

            # compute mean embeddings (output is of shape CxE)
            cluster_means = compute_cluster_means(single_input, single_target, C)

            # compute variance term, i.e. pull force
            variance_term = self._compute_variance_term(cluster_means, single_input, single_target, instance_counts,
                                                        ignore_zero_label)

            # compute unlabeled push force, i.e. push force between the mean cluster embeddings and embeddings of background pixels
            # compute only ignore_zero_label is True, i.e. a given patch contains background label
            unlabeled_push_term = 0.
            if self.unlabeled_push and contains_bg:
                unlabeled_push_term = self._compute_unlabeled_push(cluster_means, single_input, single_target)

            # compute the instance-based auxiliary loss
            instance_term = self.compute_instance_term(single_input, cluster_means, single_target)

            # compute distance term, i.e. push force
            distance_term = self._compute_distance_term(cluster_means, ignore_zero_label)

            # compute regularization term
            regularization_term = self._compute_regularizer_term(cluster_means)

            # compute total loss and sum it up
            loss = self.alpha * variance_term + \
                   self.beta * distance_term + \
                   self.gamma * regularization_term + \
                   self.instance_term_weight * instance_term + \
                   self.unlabeled_push_weight * unlabeled_push_term

            loss += loss

        # reduce across the batch dimension
        return loss.div(n_batches)


class ContrastiveLoss(AbstractContrastiveLoss):
    """
    Implementation of the standard contrastive loss defined in https://arxiv.org/pdf/1708.02551.pdf
    'Semantic Instance Segmentation with a Discriminative Loss Function', i.e. no instance-based loss term
    and no unlabeled push term.
    """

    def __init__(self, delta_var, delta_dist, norm='fro', alpha=1., beta=1., gamma=0.001):
        super(ContrastiveLoss, self).__init__(delta_var, delta_dist, norm=norm, alpha=alpha, beta=beta, gamma=gamma,
                                              unlabeled_push_weight=0., instance_term_weight=0.)

    def compute_instance_term(self, embeddings, cluster_means, target):
        # no auxiliary loss in the standard ContrastiveLoss
        return 0.


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


class AffinitySideLoss(nn.Module):
    """
    Affinity-based side loss inspired by the work of K. Lee et al.
    "Learning Dense Voxel Embeddings for 3D Neuron Reconstruction" (https://arxiv.org/pdf/1909.09872.pdf).

    Implementation provided by Constantin Pape.
    """
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


class ExtendedContrastiveLoss(AbstractContrastiveLoss):
    """
    Contrastive loss extended with the instance-based loss term and the unlabeled push term (if training done in
    semi-supervised mode).
    """

    def __init__(self, delta_var, delta_dist, norm='fro', alpha=1., beta=1., gamma=0.001,
                 unlabeled_push_weight=0.0, instance_term_weight=1.0, aux_loss='dice', pmaps_threshold=0.5, **kwargs):

        super().__init__(delta_var, delta_dist, norm=norm, alpha=alpha, beta=beta, gamma=gamma,
                         unlabeled_push_weight=unlabeled_push_weight,
                         instance_term_weight=instance_term_weight)

        # init auxiliary loss
        assert aux_loss in ['dice', 'affinity', 'dice_aff']
        if aux_loss == 'dice':
            self.aux_loss = DiceLoss(normalization='none')
        # additional auxiliary losses
        elif aux_loss == 'affinity':
            self.aux_loss = AffinitySideLoss(
                delta_dist=delta_dist,
                offset_ranges=kwargs.get('offset_ranges', [(-18, 0), (-18, 0)]),
                n_samples=kwargs.get('n_samples', 9)
            )
        elif aux_loss == 'dice_aff':
            # combine dice and affinity side loss
            dice_weight = kwargs.get('dice_weight', 1.0)
            aff_weight = kwargs.get('aff_weight', 1.0)

            dice_loss = DiceLoss(normalization='none')
            aff_loss = AffinitySideLoss(
                delta_dist=delta_dist,
                offset_ranges=kwargs.get('offset_ranges', [(-18, 0), (-18, 0)]),
                n_samples=kwargs.get('n_samples', 9)
            )

            self.aux_loss = CombinedAuxLoss(
                losses=[dice_loss, aff_loss],
                weights=[dice_weight, aff_weight]
            )

        # init dist_to_mask kernel which maps distance to the cluster center to instance probability map
        self.dist_to_mask = GaussianKernel(delta_var=self.delta_var, pmaps_threshold=pmaps_threshold)

    def _create_instance_pmaps_and_masks(self, embeddings, anchors, target):
        inst_pmaps = []
        inst_masks = []

        # permute embedding dimension at the end
        if target.dim() == 2:
            embeddings = embeddings.permute(1, 2, 0)
        else:
            embeddings = embeddings.permute(1, 2, 3, 0)

        for i, anchor_emb in enumerate(anchors):
            if i == 0:
                # ignore 0-label
                continue
            # compute distance map; embeddings is ExSPATIAL, cluster_mean is ExSINGLETON_SPATIAL, so we can just broadcast
            distance_map = torch.norm(embeddings - anchor_emb, self.norm, dim=-1)
            # convert distance map to instance pmaps and save
            inst_pmaps.append(self.dist_to_mask(distance_map).unsqueeze(0))
            # create real mask and save
            assert i in target
            inst_masks.append((target == i).float().unsqueeze(0))

        if not inst_masks:
            return None, None

        # stack along batch dimension
        inst_pmaps = torch.stack(inst_pmaps)
        inst_masks = torch.stack(inst_masks)

        return inst_pmaps, inst_masks

    def compute_instance_term(self, embeddings, cluster_means, target):
        assert embeddings.size()[1:] == target.size()
        if isinstance(self.aux_loss, AffinitySideLoss):
            return self.aux_loss(embeddings, target)
        else:
            # compute random anchors per instance
            instances = torch.unique(target)
            anchor_embeddings = []
            for i in instances:
                if i == 0:
                    # just take the mean anchor
                    anchor_embeddings.append(cluster_means[0])
                else:
                    anchor_emb = select_stable_anchor(embeddings, cluster_means[i], target == i, self.delta_var)
                    anchor_embeddings.append(anchor_emb)

            anchor_embeddings = torch.stack(anchor_embeddings, dim=0).to(embeddings.device)

            instance_pmaps, instance_masks = self._create_instance_pmaps_and_masks(embeddings, anchor_embeddings,
                                                                                   target)

            if isinstance(self.aux_loss, CombinedAuxLoss):
                return self.aux_loss(embeddings, target, instance_pmaps, instance_masks)
            else:
                if instance_masks is None:
                    return 0.
                return self.aux_loss(instance_pmaps, instance_masks).mean()


class EmbeddingConsistencyContrastiveLoss(ExtendedContrastiveLoss):
    """
    Complete loss used for SPOCO training. Essentially it extends the contrastive loss with the instance-based term,
    unlabeled push term and embedding consistency term.
    """

    def __init__(self, delta_var, delta_dist, norm='fro', alpha=1., beta=1., gamma=0.001,
                 unlabeled_push_weight=1.0, instance_term_weight=1.0, consistency_term_weight=1.0,
                 aux_loss='dice', pmaps_threshold=0.5, max_anchors=20, volume_threshold=0.05, **kwargs):

        super().__init__(delta_var, delta_dist, norm=norm, alpha=alpha, beta=beta, gamma=gamma,
                         unlabeled_push_weight=unlabeled_push_weight,
                         instance_term_weight=instance_term_weight,
                         aux_loss=aux_loss,
                         pmaps_threshold=pmaps_threshold,
                         **kwargs)

        self.consistency_term_weight = consistency_term_weight
        self.max_anchors = max_anchors
        self.volume_threshold = volume_threshold
        self.consistency_loss = DiceLoss(normalization='none')

    def __str__(self):
        return super().__str__() + f"\nconsistency_term_weight: {self.consistency_term_weight}"

    def _inst_pmap(self, emb, anchor):
        # compute distance map
        distance_map = torch.norm(emb - anchor, self.norm, dim=-1)
        # convert distance map to instance pmaps and return
        return self.dist_to_mask(distance_map)

    def emb_consistency(self, emb_q, emb_k, mask):
        inst_q = []
        inst_k = []
        for i in range(self.max_anchors):
            if mask.sum() < self.volume_threshold * mask.numel():
                break

            # get random anchor
            indices = torch.nonzero(mask, as_tuple=True)
            ind = np.random.randint(len(indices[0]))

            q_pmap = self._extract_pmap(emb_q, mask, indices, ind)
            inst_q.append(q_pmap)

            k_pmap = self._extract_pmap(emb_k, mask, indices, ind)
            inst_k.append(k_pmap)

        # stack along channel dim
        inst_q = torch.stack(inst_q)
        inst_k = torch.stack(inst_k)

        loss = self.consistency_loss(inst_q, inst_k)
        return loss

    def _extract_pmap(self, emb, mask, indices, ind):
        if mask.dim() == 2:
            y, x = indices
            anchor = emb[:, y[ind], x[ind]]
            emb = emb.permute(1, 2, 0)
        else:
            z, y, x = indices
            anchor = emb[:, z[ind], y[ind], x[ind]]
            emb = emb.permute(1, 2, 3, 0)

        return self._inst_pmap(emb, anchor)

    def forward(self, input, target):
        assert len(input) == 2
        emb_q, emb_k = input

        # compute extended contrastive loss only on the embeddings coming from q
        contrastive_loss = super().forward(emb_q, target)

        # compute consistency term
        for e_q, e_k, t in zip(emb_q, emb_k, target):
            unlabeled_mask = (t == 0).int()
            if unlabeled_mask.sum() < self.volume_threshold * unlabeled_mask.numel():
                continue
            emb_consistency_loss = self.emb_consistency(e_q, e_k, unlabeled_mask)
            contrastive_loss += self.consistency_term_weight * emb_consistency_loss

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


def create_loss(delta_var, delta_dist,
                alpha, beta, gamma,
                unlabeled_push_weight, instance_term_weight,
                consistency_weight, kernel_threshold):
    """
    Creates an instance of the embedding loss based on the parameters provided.
    If `unlabeled_push_weight` is set to zero ContrastiveLos or ExtendedContrastiveLoss (if `instance_term_weight`
    is greater than 0) is returned, otherwise the EmbeddingConsistencyContrastiveLoss is returned.

    Args:
        delta_var: pull force hinge
        delta_dist: push force hinge
        alpha: pull force weight
        beta: push force weight
        gamma: regularizer term weight
        unlabeled_push_weight: unlabeled push force weight
        instance_term_weight: instance-based loss weight
        consistency_weight: embedding consistency loss weight
        kernel_threshold: threshold used for the differentiable instance selection

    Returns:
        nn.Module: an instance of the loss function

    """
    assert delta_var > 0
    assert delta_dist > 0

    if unlabeled_push_weight == 0:
        # no unlabeled region, so it's a standard (Extended)ContrastiveLoss
        if instance_term_weight > 0:
            return ExtendedContrastiveLoss(delta_var=delta_var, delta_dist=delta_dist,
                                           alpha=alpha, beta=beta, gamma=gamma,
                                           unlabeled_push_weight=0.,
                                           instance_term_weight=instance_term_weight,
                                           aux_loss='dice', pmaps_threshold=kernel_threshold)
        else:
            return ContrastiveLoss(delta_var=delta_var, delta_dist=delta_dist,
                                   alpha=alpha, beta=beta, gamma=gamma)
    else:
        # unlabeled push weight defined: 0-label corresponds to the unlabeled region, we're in a sparse setting
        return EmbeddingConsistencyContrastiveLoss(delta_var=delta_var, delta_dist=delta_dist,
                                                   alpha=alpha, beta=beta, gamma=gamma,
                                                   unlabeled_push_weight=unlabeled_push_weight,
                                                   instance_term_weight=instance_term_weight,
                                                   consistency_term_weight=consistency_weight,
                                                   aux_loss='dice', pmaps_threshold=kernel_threshold)
