import math
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

SUPPORTED_DATASETS = ['cvppp', 'cityscapes', 'ovules', 'mitoem', 'stem']


class GaussianKernel(nn.Module):
    """
    A kernel function used to convert the distance map (i.e. `||embeddings - anchor_embedding||`)
    into an instance probability map. It can be interpreted as a probability that an `anchor_embedding` belongs
    to a given region of the image. The kernel is a Gaussian function with a fixed variance.

    Args
        delta_var (float): pull force distance margin
        pmaps_threshold (float): kernel value for embeddings delta_var away from the anchor
    """

    def __init__(self, delta_var, pmaps_threshold):
        super().__init__()
        self.delta_var = delta_var
        # dist_var^2 = -2*sigma*ln(pmaps_threshold)
        self.two_sigma = delta_var * delta_var / (-math.log(pmaps_threshold))

    def forward(self, dist_map):
        return torch.exp(- dist_map * dist_map / self.two_sigma)


def create_optimizer(model, lr, wd=0., betas=(0.9, 0.999)):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module

    betas = tuple(betas)
    return optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)


def create_lr_scheduler(optimizer, patience, mode, factor=0.2):
    lr_scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    return lr_scheduler


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


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename):
    checkpoint_dir, _ = os.path.split(filename)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    torch.save(state, filename)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(filename, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict',
                    map_location=None):
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


def pca_project(embeddings):
    """
    Project embeddings into 3-dim RGB space for visualization purposes

    Args:
        embeddings: ExSpatial embedding tensor

    Returns:
        RGB image
    """
    assert embeddings.ndim == 3
    # reshape (C, H, W) -> (C, H * W) and transpose
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose()
    # init PCA with 3 principal components: one for each RGB channel
    pca = PCA(n_components=3)
    # fit the model with embeddings and apply the dimensionality reduction
    flattened_embeddings = pca.fit_transform(flattened_embeddings)
    # reshape back to original
    shape = list(embeddings.shape)
    shape[0] = 3
    img = flattened_embeddings.transpose().reshape(shape)
    # normalize to [0, 255]
    img = 255 * (img - np.min(img)) / np.ptp(img)
    return img.astype('uint8')


def minmax_norm(img):
    channels = [np.nan_to_num((c - np.min(c)) / np.ptp(c)) for c in img]
    return np.stack(channels, axis=0)
