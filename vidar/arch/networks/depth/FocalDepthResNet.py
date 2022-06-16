# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

from vidar.arch.blocks.depth.SigmoidToInvDepth import SigmoidToInvDepth
from vidar.arch.networks.BaseNet import BaseNet
from vidar.arch.networks.decoders.DepthDecoder import DepthDecoder
from vidar.arch.networks.encoders.ResNetEncoder import ResNetEncoder as ResnetEncoder
from vidar.utils.depth import inv2depth, depth2inv


def allow_multicam_input(func):
    """Decorator to allow 5-dim input tensors by reshaping, e.g. BxNxCxHxW -> {BxN}xCxHxW"""
    # TODO(soonminh): this decorator works only for below network.
    # TODO(soonminh): should be improved and moved to vidar/utils/decorator.py
    def inner(cls, rgb, intrinsics, **kwargs):
        B = N = None
        if rgb.dim() == 5:
            B, N = rgb.shape[:2]
            rgb = rgb.view(B * N, *rgb.shape[2:])
            intrinsics = intrinsics.view(B * N, *intrinsics.shape[2:])

        out = func(cls, rgb, intrinsics, **kwargs)
        if B is not None:
            # HACK(soonminh): Assuming a specific output type; a dict of list of tensor
            out = {k: [v.view(B, N, *v.shape[1:]) for v in vs] for k, vs in out.items()}
        return out
    return inner


class FocalDepthResNet(BaseNet, ABC):
    """
    Depth network with focal length normalization

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.networks['encoder'] = ResnetEncoder(cfg.encoder)
        cfg.decoder.num_ch_enc = self.networks['encoder'].num_ch_enc
        self.networks['decoder'] = DepthDecoder(cfg.decoder)
        self.scale_inv_depth = SigmoidToInvDepth(
            min_depth=cfg.min_depth, max_depth=cfg.max_depth)

    @allow_multicam_input
    def forward(self, rgb, intrinsics, **kwargs):
        """Network forward pass"""

        # TODO(sohwang): allow 5-dim input tensors, need to write a decorator
        x = self.networks['encoder'](rgb)
        x = self.networks['decoder'](x)
        inv_depths = [x[('output', i)] for i in range(4)]

        if self.training:
            inv_depths = [self.scale_inv_depth(inv_depth)[0] for inv_depth in inv_depths]
        else:
            inv_depths = [self.scale_inv_depth(inv_depths[0])]

        depths = inv2depth(inv_depths)
        depths = [d * intrinsics[:, 0, 0].view(rgb.shape[0], 1, 1, 1) for d in depths]
        inv_depths = depth2inv(depths)

        return {
            'inv_depths': inv_depths
        }
