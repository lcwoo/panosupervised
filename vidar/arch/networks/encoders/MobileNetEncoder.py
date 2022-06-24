import timm
from abc import ABC
from timm.models.features import FeatureHooks

import torch
import torch.nn as nn

TIMM_LIGHTWEIGHT_MODELS = (
   'mixnet_m',          # 13.90 ms
   'mixnet_s',          #  9.33 ms
   'efficientnet_b0',   #  7.62 ms
   'mnasnet_a1',        #  5.07 ms
   'spnasnet_100',      #  4.90 ms
   'mobilenetv2_100',   #  4.09 ms
   'efficientnet_es',   #  4.08 ms
   'fbnetc_100',        #  4.00 ms,
   'lcnet_100',         #  2.70 ms,    mobilenetv3.py,   https://arxiv.org/pdf/2109.15099.pdf
)


class MobileNetEncoder(nn.Module, ABC):
    """
    MobileNet encoder network

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__()

        # TODO(soonminh): support more lightweight models
        assert cfg.version in ['lcnet_100'], f'(WIP) This model is not tested yet: {cfg.version}'

        # Should move into encoder networks
        assert cfg.version in TIMM_LIGHTWEIGHT_MODELS, f'Unknown networks: {cfg.version}'

        # model = timm.create_model(cfg.version, pretrained=cfg.pretrained)
        model = timm.create_model(cfg.version, pretrained=cfg.pretrained, features_only=True)

        # Update feature hooks: overwirte
        hooks = [{**info, 'hook_type': 'forward'} for info in model.feature_info.info]
        model.feature_hooks = FeatureHooks(hooks, model.named_modules())

        self.model_info = model.default_cfg
        self.register_buffer('input_mean', torch.FloatTensor(self.model_info['mean']).view(1, 3, 1, 1))
        self.register_buffer('input_str', torch.FloatTensor(self.model_info['std']).view(1, 3, 1, 1))
        self.model = model

        self.reduction = model.feature_info.reduction()
        self.num_ch_enc = model.feature_info.channels()

    def forward(self, input_image):
        x = (input_image - self.input_mean) / self.input_str
        features = self.model(x)
        return features
