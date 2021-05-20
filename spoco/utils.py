import io
import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

plt.ioff()
plt.switch_backend('agg')

SUPPORTED_DATASETS = ['cvppp', 'dsb', 'ovules', 'mitoem', 'stem']


class GaussianKernel(nn.Module):
    def __init__(self, delta_var, pmaps_threshold):
        super().__init__()
        self.delta_var = delta_var
        # dist_var^2 = -2*sigma*ln(pmaps_threshold)
        self.two_sigma = delta_var * delta_var / (-math.log(pmaps_threshold))

    def forward(self, dist_map):
        return torch.exp(- dist_map * dist_map / self.two_sigma)


def create_optimizer(lr, model, wd=0., betas=(0.9, 0.999)):
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    return optimizer


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


def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)


class _TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, is3d=True, **kwargs):
        self.is3d = is3d

    def __call__(self, name, batch):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert img.ndim == 2 or img.ndim == 3, 'Only 2D (HW) and 3D (CHW) images are accepted for display'

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert C == 1 or C == 3, 'Only (1, H, W) or (3, H, W) images are supported'

            return tag, img

        tagged_images = self.process_batch(name, batch)

        return list(map(_check_img, tagged_images))

    def process_batch(self, name, batch):
        raise NotImplementedError


class DefaultTensorboardFormatter(_TensorboardFormatter):
    def __init__(self, skip_last_target=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_last_target = skip_last_target

    def process_batch(self, name, batch):
        if name == 'targets' and self.skip_last_target:
            batch = batch[:, :-1, ...]

        tagged_images = []

        if self.is3d:
            tag_template = '{}/batch_{}/channel_{}/slice_{}'
            if batch.ndim == 5:
                # NCDHW
                slice_idx = batch.shape[2] // 2  # get the middle slice
                for batch_idx in range(batch.shape[0]):
                    for channel_idx in range(batch.shape[1]):
                        tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                        img = batch[batch_idx, channel_idx, slice_idx]
                        tagged_images.append((tag, self._normalize_img(img)))
            else:
                # NDHW
                slice_idx = batch.shape[1] // 2  # get the middle slice
                for batch_idx in range(batch.shape[0]):
                    tag = tag_template.format(name, batch_idx, 0, slice_idx)
                    img = batch[batch_idx, slice_idx]
                    tagged_images.append((tag, self._normalize_img(img)))
        else:
            tag_template = '{}/batch_{}/channel_{}'
            if batch.ndim == 4:
                # NCHW
                for batch_idx in range(batch.shape[0]):
                    for channel_idx in range(batch.shape[1]):
                        tag = tag_template.format(name, batch_idx, channel_idx)
                        img = batch[batch_idx, channel_idx]
                        tagged_images.append((tag, self._normalize_img(img)))
            else:
                # NHW
                for batch_idx in range(batch.shape[0]):
                    tag = tag_template.format(name, batch_idx, 0)
                    img = batch[batch_idx]
                    tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))


def _find_masks(batch, min_size=10):
    """Center the z-slice in the 'middle' of a given instance, given a batch of instances

    Args:
        batch (ndarray): 5d numpy tensor (NCDHW)
    """
    assert batch.ndim == 5
    result = []
    for b in batch:
        assert b.shape[0] == 1
        patch = b[0]
        z_sum = patch.sum(axis=(1, 2))
        coords = np.where(z_sum > min_size)[0]
        if len(coords) > 0:
            ind = coords[len(coords) // 2]
            result.append(b[:, ind:ind + 1, ...])
        else:
            ind = b.shape[1] // 2
            result.append(b[:, ind:ind + 1, ...])

    return np.stack(result, axis=0)


class EmbeddingsTensorboardFormatter(DefaultTensorboardFormatter):
    def __init__(self, plot_variance=False, **kwargs):
        super().__init__(**kwargs)
        self.plot_variance = plot_variance

    def process_batch(self, name, batch):
        if name.startswith('predictions'):
            return self._embeddings_to_rgb(name, batch)
        elif name == 'real_masks' or name == 'fake_masks':
            if self.is3d:
                # find proper z-slice
                batch = _find_masks(batch)

            return super().process_batch(name, batch)
        else:
            return super().process_batch(name, batch)

    def _embeddings_to_rgb(self, name, batch):
        tagged_images = []

        for batch_idx in range(batch.shape[0]):
            if self.is3d:
                tag_template = name + '/batch_{}/slice_{}'
                slice_idx = batch.shape[2] // 2  # get the middle slice
                tag = tag_template.format(batch_idx, slice_idx)
                img = batch[batch_idx, :, slice_idx, ...]  # CHW
            else:
                tag_template = name + '/batch_{}'
                tag = tag_template.format(batch_idx)
                img = batch[batch_idx]  # CHW

            # get the PCA projection
            rgb_img = pca_project(img)
            tagged_images.append((tag, rgb_img))
            if self.plot_variance:
                cum_explained_variance_img = self._plot_cum_explained_variance(img)
                tagged_images.append((f'cumulative_explained_variance/batch_{batch_idx}', cum_explained_variance_img))

        return tagged_images

    def _plot_cum_explained_variance(self, embeddings):
        # reshape (C, H, W) -> (C, H * W) and transpose
        flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).transpose()
        # fit PCA to the data
        pca = PCA().fit(flattened_embeddings)

        plt.clf()
        # plot cumulative explained variance ratio
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        img = np.asarray(Image.open(buf)).transpose(2, 0, 1)
        return img


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


def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    print(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        print(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
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
