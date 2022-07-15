import numpy as np
from abc import ABC
from collections import OrderedDict

import torch
import torch.nn as nn

from vidar.arch.networks.layers.convs import ConvBlock, Conv3x3, upsample
from vidar.arch.networks.layers.transformer import CVT
from vidar.utils.config import cfg_has


def has_same_res(x, y):
    return x.shape[-2:] == y.shape[-2:]


class SurroundDepthDecoder(nn.Module, ABC):
    """
    SurroundDepth decoder network

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.fusion_skip = cfg_has(cfg, 'fusion_skip', True)

        self.num_output_channels = cfg_has(cfg, 'num_output_channels', 1)
        self.upsample_mode = cfg_has(cfg, 'upsample_mode', 'nearest')
        self.num_scales = cfg_has(cfg, 'num_scales', 4)
        self.scales = range(self.num_scales)
        self.use_skips = cfg_has(cfg, 'use_skips', True)

        self.iter_num = [8, 8, 8, 8, 8]
        self.num_ch_enc = cfg.num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.setup_decoder()

    def setup_decoder(self):
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.cross = {}

        for i in range(len(self.num_ch_enc)):
            self.cross[i] = CVT(input_channel=self.num_ch_enc[i], downsample_ratio=2**(len(self.num_ch_enc) -1 - i), iter_num=self.iter_num[i])

        self.decoder_cross = nn.ModuleList(list(self.cross.values()))


    def forward(self, input_features, meta):
        self.outputs = {}
        for i in range(len(input_features)):
            B, C, H, W = input_features[i].shape
            if self.fusion_skip:
                input_features[i] = input_features[i] + self.cross[i](input_features[i].reshape(-1, 6, C, H, W)).reshape(B, C, H, W)
            else:
                input_features[i] = self.cross[i](input_features[i].reshape(-1, 6, C, H, W)).reshape(B, C, H, W)

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
