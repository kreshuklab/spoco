import numpy as np
import torch
import torch.nn as nn

from spoco.buildingblocks import create_encoders, create_decoders, DoubleConv


class AbstractUNet(nn.Module):
    """
    Base class for the U-Net architecture.

    Args:
        in_channels (int): number of input channels
        out_channels (int): embedding dimensionality
        f_maps (int, tuple): number of feature maps at each level of the encoder. In case of an integer
            the number of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4.
            If it's a tuple, the number of feature maps is given by the tuple elements, e.g. [32, 64, 128, 256, 512]
        layer_order (string): determines the order of layers in a `SingleConv` module.
            E.g. 'gcr' stands for GroupNorm+Conv+ReLU, 'bcr' stands for BatchNorm+Conv+ReLU.
            See `SingleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        conv_kernel_size (int or tuple): size of the convolving kernel in the convolutions
        pool_kernel_size (int or tuple): the size of the pooling window
        conv_padding (int or tuple): add zero-padding to all three sides of the input
        is3d (bool): is it 2D or 3D U-Net
    """

    def __init__(self, in_channels, out_channels, f_maps, layer_order='bcr', num_groups=8, conv_kernel_size=3,
                 pool_kernel_size=2, conv_padding=1, is3d=True):
        super(AbstractUNet, self).__init__()

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, DoubleConv, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size, is3d)

        # create decoder path
        self.decoders = create_decoders(f_maps, DoubleConv, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        is3d)

        # in the last layer a 1×1 convolution reduces the number of output channels to the embedding dimension
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output (i.e. the U-Net's bottleneck) from the list
        # REMEMBER: it's the 1st one in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output of the previous decoder
            x = decoder(encoder_features, x)

        # return the logits as the final pixel embeddings
        return self.final_conv(x)


class UNet3D(AbstractUNet):
    """
    3DUnet model from
    "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>.

    Uses nearest neighbor upsampling + Conv in the decoder, so called UpConvolution instead of transposed convolutions.
    """

    def __init__(self, in_channels, out_channels, f_maps, layer_order='bcr', num_groups=8, conv_padding=1):
        super(UNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels, f_maps=f_maps,
                                     layer_order=layer_order, num_groups=num_groups, conv_padding=conv_padding,
                                     is3d=True)


class UNet2D(AbstractUNet):
    """
    A standard 2D UNet: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        <https://arxiv.org/abs/1505.04597>.

    Uses nearest neighbor upsampling + Conv in the decoder, so called UpConvolution instead of transposed convolutions.
    """

    def __init__(self, in_channels, out_channels, f_maps, layer_order='bcr', num_groups=8, conv_padding=1):
        super(UNet2D, self).__init__(in_channels=in_channels, out_channels=out_channels, f_maps=f_maps,
                                     layer_order=layer_order, num_groups=num_groups, conv_padding=conv_padding,
                                     is3d=False)


class SpocoNet(nn.Module):
    """
    Wrapper around the f-network and the moving average g-network.

    Args:
        net_f (nn.Module): embedding network that is trained using SPOCO loss
        net_g (nn.Module): embedding network implemented as an exponential moving average of the net_f weights
        m (float): momentum parameter for the moving average
    """

    def __init__(self, net_f, net_g, m=0.999):
        super(SpocoNet, self).__init__()

        self.net_f = net_f
        self.net_g = net_g
        self.m = m

        # initialize g weights to be equal to f weights
        for param_f, param_g in zip(self.net_f.parameters(), self.net_g.parameters()):
            param_g.data.copy_(param_f.data)  # initialize
            param_g.requires_grad = False  # freeze g parameters

    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update of the g-network parameters.
        """
        for param_f, param_g in zip(self.net_f.parameters(), self.net_g.parameters()):
            param_g.data = param_g.data * self.m + param_f.data * (1. - self.m)

    def forward(self, im_f, im_g):
        # compute f-embeddings
        emb_f = self.net_f(im_f)

        # compute g-embeddings
        with torch.no_grad():  # no gradient to g-embeddings
            self._momentum_update()  # momentum update of g
            emb_g = self.net_g(im_g)

        return emb_f, emb_g


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def create_model(args):
    assert args.model_name in ["UNet2D", "UNet3D"]
    if args.model_name == "UNet2D":
        model_class = UNet2D
    else:
        model_class = UNet3D

    net_f = model_class(
        in_channels=args.model_in_channels,
        out_channels=args.model_out_channels,
        f_maps=args.model_feature_maps,
        layer_order=args.model_layer_order
    )

    if not args.spoco:
        return net_f

    net_g = model_class(
        in_channels=args.model_in_channels,
        out_channels=args.model_out_channels,
        f_maps=args.model_feature_maps,
        layer_order=args.model_layer_order
    )

    return SpocoNet(net_f, net_g, m=args.momentum)
