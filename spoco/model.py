import numpy as np
import torch
import torch.nn as nn

from spoco.buildingblocks import create_encoders, create_decoders, DoubleConv


class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'gcr' stands for GroupNorm+Conv+ReLU.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        is3d (bool): is it 2d or 3d UNet
    """

    def __init__(self, in_channels, out_channels, basic_module, f_maps, layer_order='bcr',
                 num_groups=8, conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, is3d=True):
        super(AbstractUNet, self).__init__()

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size, is3d)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        is3d)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
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

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        return x


class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, f_maps, layer_order='bcr', num_groups=8, conv_padding=1):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     conv_padding=conv_padding,
                                     is3d=True)


class UNet2D(AbstractUNet):
    """
    Just a standard 2D UNet
    """

    def __init__(self, in_channels, out_channels, f_maps, layer_order='bcr',
                 num_groups=8, conv_padding=1):
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     conv_padding=conv_padding,
                                     is3d=False)


class SpocoUNet(nn.Module):
    def __init__(self, net_f, net_g, m=0.999, init_equal=True):
        super(SpocoUNet, self).__init__()

        self.net_f = net_f
        self.net_g = net_g
        self.m = m

        if init_equal:
            # initialize g weights to be equal to f weights
            for param_f, param_g in zip(self.net_f.parameters(), self.net_g.parameters()):
                param_g.data.copy_(param_f.data)  # initialize
                param_g.requires_grad = False  # freeze g parameters

    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update of the g
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
    if hasattr(args, 'momentum'):
        momentum = args.momentum
    else:
        momentum = 0.999

    return SpocoUNet(net_f, net_g, m=momentum)
