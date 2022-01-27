import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import ImageFilter
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from skimage import measure


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 2D, 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, axis_prob=0.5, channelwise=False, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axis_prob = axis_prob
        self.channelwise = channelwise

    def __call__(self, m):
        assert m.ndim in [2, 3, 4]
        if self.channelwise:
            axes = range(m.ndim - 1)
        else:
            axes = range(m.ndim)

        for axis in axes:
            if self.random_state.uniform() > self.axis_prob:
                if self.channelwise:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)
                else:
                    m = np.flip(m, axis)

        return m


class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state, channelwise=False, **kwargs):
        self.random_state = random_state
        self.channelwise = channelwise

    def _rot_axis(self, ndim):
        if ndim == 3:
            # always rotate 3d volume around z-axis
            return 1, 2
        else:
            return 0, 1

    def __call__(self, m):
        assert m.ndim in [2, 3, 4]

        # pick number of rotations at random
        k = self.random_state.randint(0, 4)

        if self.channelwise:
            axis = self._rot_axis(m.ndim - 1)
            channels = [np.rot90(m[c], k, axis) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)
        else:
            axis = self._rot_axis(m.ndim)
            m = np.rot90(m, k, axis)

        return m


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=30, mode='reflect', order=0, channelwise=False, **kwargs):
        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.mode = mode
        self.order = order
        self.channelwise = channelwise

    def _rot_axis(self, ndim):
        if ndim == 3:
            # always rotate 3d volume around z-axis
            return 2, 1
        else:
            return 1, 0

    def __call__(self, m):
        assert m.ndim in [2, 3, 4]

        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if self.channelwise:
            axis = self._rot_axis(m.ndim - 1)
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
                        for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)
        else:
            axis = self._rot_axis(m.ndim)
            m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)

        return m


class RandomContrast:
    """
    Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
    """

    def __init__(self, random_state, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1, **kwargs):
        self.random_state = random_state
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
            result = self.mean + alpha * (m - self.mean)
            return np.clip(result, -1, 1)

        return m


# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=0 when transforming the labels
class ElasticDeformation:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, random_state, spline_order, alpha=2000, sigma=50, execution_probability=0.1, channelwise=False,
                 **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        :param channelwise: if True treat 1st dimension as channel dimension and apply deformations for each channel individually
        """
        self.random_state = random_state
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability
        self.channelwise = channelwise

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            assert m.ndim in [2, 3, 4]

            if self.channelwise:
                volume_shape = m.shape[1:]
            else:
                volume_shape = m.shape

            d_array = [
                gaussian_filter(
                    self.random_state.randn(*volume_shape),
                    self.sigma,
                    mode="reflect"
                ) * self.alpha for _ in range(len(volume_shape))
            ]

            if len(volume_shape) == 3:
                z_dim, y_dim, x_dim = volume_shape
                z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
                dz, dy, dx = d_array
                indices = z + dz, y + dy, x + dx
            else:
                y_dim, x_dim = volume_shape
                y, x = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing='ij')
                dy, dx = d_array
                indices = y + dy, x + dx

            if self.channelwise:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode='reflect') for c in m]
                return np.stack(channels, axis=0)
            else:
                return map_coordinates(m, indices, order=self.spline_order, mode='reflect')

        return m


class CropToFixed:
    def __init__(self, random_state, size=(256, 256), centered=False, channelwise=False, **kwargs):
        self.random_state = random_state
        self.crop_y, self.crop_x = size
        self.centered = centered
        self.channelwise = channelwise

    def __call__(self, m):
        def _padding(pad_total):
            half_total = pad_total // 2
            return (half_total, pad_total - half_total)

        def _rand_range_and_pad(crop_size, max_size):
            """
            Returns a tuple:
                max_value (int) for the corner dimension. The corner dimension is chosen as `self.random_state(max_value)`
                pad (int): padding in both directions; if crop_size is lt max_size the pad is 0
            """
            if crop_size < max_size:
                return max_size - crop_size, (0, 0)
            else:
                return 1, _padding(crop_size - max_size)

        def _start_and_pad(crop_size, max_size):
            if crop_size < max_size:
                return (max_size - crop_size) // 2, (0, 0)
            else:
                return 0, _padding(crop_size - max_size)

        assert m.ndim in (2, 3, 4)
        y, x = m.shape[-2:]

        if not self.centered:
            y_range, y_pad = _rand_range_and_pad(self.crop_y, y)
            x_range, x_pad = _rand_range_and_pad(self.crop_x, x)

            y_start = self.random_state.randint(y_range)
            x_start = self.random_state.randint(x_range)

        else:
            y_start, y_pad = _start_and_pad(self.crop_y, y)
            x_start, x_pad = _start_and_pad(self.crop_x, x)

        if self.channelwise:
            channels = []
            for c in range(m.shape[0]):
                result = m[c, y_start:y_start + self.crop_y, x_start:x_start + self.crop_x]
                if result.ndim == 3:
                    pad_width = ((0, 0), y_pad, x_pad)
                else:
                    pad_width = (y_pad, x_pad)
                channels.append(np.pad(result, pad_width=pad_width, mode='reflect'))
            return np.stack(channels, axis=0)
        else:
            result = m[y_start:y_start + self.crop_y, x_start:x_start + self.crop_x]
            if result.ndim == 3:
                pad_width = ((0, 0), y_pad, x_pad)
            else:
                pad_width = (y_pad, x_pad)
            return np.pad(result, pad_width=pad_width, mode='reflect')


class Standardize:
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    """

    def __init__(self, eps=1e-10, mean=None, std=None, channelwise=False, **kwargs):
        if mean is not None or std is not None:
            assert mean is not None and std is not None
        self.mean = mean
        self.std = std
        self.eps = eps
        self.channelwise = channelwise

    def __call__(self, m):
        if self.mean is not None:
            mean, std = self.mean, self.std
        else:
            if self.channelwise:
                # normalize per-channel
                axes = list(range(m.ndim))
                # average across channels
                axes = tuple(axes[1:])
                mean = np.mean(m, axis=axes, keepdims=True)
                std = np.std(m, axis=axes, keepdims=True)
            else:
                mean = np.mean(m)
                std = np.std(m)

        return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)


class PercentileNormalizer:
    def __init__(self, pmin, pmax, channelwise=False, eps=1e-10, **kwargs):
        self.eps = eps
        self.pmin = pmin
        self.pmax = pmax
        self.channelwise = channelwise

    def __call__(self, m):
        if self.channelwise:
            axes = list(range(m.ndim))
            # average across channels
            axes = tuple(axes[1:])
            pmin = np.percentile(m, self.pmin, axis=axes, keepdims=True)
            pmax = np.percentile(m, self.pmax, axis=axes, keepdims=True)
        else:
            pmin = np.percentile(m, self.pmin)
            pmax = np.percentile(m, self.pmax)

        return (m - pmin) / (pmax - pmin + self.eps)


class Normalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, min_value, max_value, **kwargs):
        assert max_value > min_value
        self.min_value = min_value
        self.value_range = max_value - min_value

    def __call__(self, m):
        norm_0_1 = (m - self.min_value) / self.value_range
        return np.clip(2 * norm_0_1 - 1, -1, 1)


class AdditiveGaussianNoise:
    def __init__(self, random_state, scale=(0.0, 1.0), execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.scale = scale

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            std = self.random_state.uniform(self.scale[0], self.scale[1])
            gaussian_noise = self.random_state.normal(0, std, size=m.shape)
            return m + gaussian_noise
        return m


class AdditivePoissonNoise:
    def __init__(self, random_state, lam=(0.0, 1.0), execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.lam = lam

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            lam = self.random_state.uniform(self.lam[0], self.lam[1])
            poisson_noise = self.random_state.poisson(lam, size=m.shape)
            return m + poisson_noise
        return m


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when expand_dims=True.
    """

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [2, 3, 4]
        # add channel dimension
        if self.expand_dims:
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))


class Relabel:
    """
    Relabel a numpy array of labels into a consecutive numbers, e.g.
    [10, 10, 0, 6, 6] -> [2, 2, 0, 1, 1]. Useful when one has an instance segmentation volume
    at hand and would like to create a one-hot-encoding for it. Without a consecutive labeling the task would be harder.
    """

    def __init__(self, append_original=False, run_cc=True, ignore_label=None, **kwargs):
        self.append_original = append_original
        self.ignore_label = ignore_label
        self.run_cc = run_cc

        if ignore_label is not None:
            assert append_original, "ignore_label present, so append_original must be true, so that one can localize the ignore region"

    def __call__(self, m):
        orig = m
        m = np.array(m)

        if self.run_cc:
            # assign 0 to the ignore region
            m = measure.label(m, background=self.ignore_label)

        _, unique_labels = np.unique(m, return_inverse=True)
        result = unique_labels.reshape(m.shape)
        if self.append_original:
            result = np.stack([result, orig])
        return result


class Identity:
    def __init__(self, **kwargs):
        pass

    def __call__(self, m):
        return m


class RgbToLabel:
    def __call__(self, img):
        img = np.array(img)
        assert img.ndim == 3 and img.shape[2] == 3
        result = img[..., 0] * 65536 + img[..., 1] * 256 + img[..., 2]
        return result


class LabelToTensor:
    def __init__(self, is_semantic):
        self.is_semantic = is_semantic

    def __call__(self, m):
        m = np.array(m)
        result = torch.from_numpy(m.astype(dtype='int64'))
        if self.is_semantic:
            result = result > 0
            result = result.float()
        return result


class ImgNormalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if self.mean is None:
            mean = torch.mean(tensor, dim=(1, 2))
            std = torch.std(tensor, dim=(1, 2))
        else:
            mean = self.mean
            std = self.std

        return F.normalize(tensor, mean, std)


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
