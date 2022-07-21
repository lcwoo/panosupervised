# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from vidar.arch.networks.layers.convs import ConvBlock, Conv3x3, upsample, downsample
from vidar.arch.networks.layers.panodepth import multicam_fusion
from vidar.utils.config import cfg_has
from vidar.utils.viz import viz_photo


def has_same_res(x, y):
    return x.shape[-2:] == y.shape[-2:]


class PanoDepthDecoder(nn.Module, ABC):
    """
    PanoDepth decoder network

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__()

        self.input_cameras = cfg.input_cameras

        self.num_scales = cfg_has(cfg, 'num_scales', 4)
        self.use_skips = cfg.use_skips

        # Channels after camera fusion
        self.num_ch_mid = [os[0] for _, _, _, os in cfg.scale_and_shapes[self.input_cameras[0]]]
        # Decoder feature channels
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.num_ch_out = cfg.num_ch_out

        ### 1. Multi-camera fusion modules
        # depth_hypothesis = cfg_has(cfg, 'depth_hypothesis', [2, 3, 5, 10, 20, 50, 100, 200])
        depth_hypothesis = cfg_has(cfg, 'depth_hypothesis', [3, 10, 30, 90])

        # TODO(soonminh): support more fusion modules or make it configurable for ablation study
        fusion_module = getattr(multicam_fusion, cfg.fusion_type, 'MultiDepthSweepFusion')
        camera_fusion = []
        for i in range(self.num_scales + 1):
            shapes_scale = {cam: shapes[i] for cam, shapes in cfg.scale_and_shapes.items()}
            camera_fusion.append(fusion_module(shapes_scale, depth_hypothesis))
        self.camera_fusions = nn.ModuleList(camera_fusion)

        ### 2. Decoder
        self.setup_decoder()
        self.downsample = cfg_has(cfg, 'downsample', False)

        ### 3. Activation
        if cfg.activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif cfg.activation == 'identity':
            self.activation = nn.Identity()
        elif cfg.activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            raise ValueError('Invalid activation function {}'.format(cfg.activation))

    def setup_decoder(self):
        self.convs = OrderedDict()
        for i in range(self.num_scales, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_mid[-1] if i == self.num_scales else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_mid[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for i in range(self.num_scales):
            self.convs[('outconv', i)] = Conv3x3(self.num_ch_dec[i], self.num_ch_out)

        self.decoder = nn.ModuleList(list(self.convs.values()))


    def forward(self, input_features, meta, return_logs=False):
        """Network forward pass"""
        # TODO(soonminh): return features for analysis/debugging
        input_agg_feats = []
        for i in range(self.num_scales + 1):
            features_scale = {cam: cam_feats[i] for cam, cam_feats in input_features.items()}
            agg_feats = self.camera_fusions[i](features_scale, meta, return_logs)
            input_agg_feats.append(agg_feats)

        outputs = {'log_images': {}}

        if return_logs:
            outputs['log_images'] = {'input_agg_feats': []}
            for _feat in input_agg_feats:
                norm = torch.linalg.norm(_feat.detach(), 2, dim=1, keepdim=True)
                norm /= norm.max()
                outputs['log_images']['input_agg_feats'].append(
                    (viz_photo(norm[0]) * 255.0).astype(np.uint8)
                )

        x = input_agg_feats[-1]
        for i in range(self.num_scales, -1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)] if not has_same_res(x, input_agg_feats[i-1]) else [x]
            if self.use_skips and i > 0:
                x += [input_agg_feats[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
            if i in range(self.num_scales):
                # HACK(soonminh): save memory & computation
                feat = downsample(x) if self.downsample else x
                outputs[('features', i)] = feat
                outputs[('output', i)] = self.activation(
                    self.convs[('outconv', i)](feat))

        return outputs
