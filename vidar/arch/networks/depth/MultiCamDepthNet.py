# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch
import torch.nn as nn

from vidar.arch.blocks.depth.SigmoidToInvDepth import SigmoidToInvDepth
from vidar.arch.networks.BaseNet import BaseNet
from vidar.utils.config import cfg_has, get_folder_name, load_class

_KNOWN_DECODER_TYPES = (
    'PanoDepthDecoder',             # Proposed
)


class MultiCamDepthNet(BaseNet, ABC):
    """
    Multi-camera Depth network

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.input_cameras = [c for c in cfg.encoders.keys() if c.startswith('camera')]
        self.freeze_encoders = cfg_has(cfg, 'freeze_encoders', False)

        ### Encoders
        scale_and_shapes = dict()
        out_shape = cfg_has(cfg.decoder, 'out_shape', (128, 1024))

        self.networks['encoders'] = nn.ModuleDict()
        for camera in self.input_cameras:
            cfg_per_cam = cfg.encoders.dict[camera]
            file = cfg_has(cfg_per_cam, 'file', 'MobileNetEncoder')
            folder, name = get_folder_name(file, 'networks')
            encoder_module = load_class(name, folder)(cfg_per_cam)
            self.networks['encoders'][camera] = encoder_module

            in_shape = cfg_has(cfg_per_cam, 'in_shape', (384, 640))
            # scale_and_shapes[camera] = self._get_output_shape(encoder_module, in_shape, out_shape)
            scale_and_shapes[camera] = [(s, (ch, in_shape[0]//s, in_shape[1]//s), (ch, out_shape[0]//s, out_shape[1]//s))
                for s, ch in zip(encoder_module.reduction, encoder_module.num_ch_enc)]

        cfg.decoder.scale_and_shapes = scale_and_shapes
        cfg.decoder.input_cameras = self.input_cameras

        ### Decoder
        folder, name = get_folder_name(cfg.decoder.file, 'networks')
        assert name in _KNOWN_DECODER_TYPES, f'Unknown decoder type: {name}'
        self.networks['decoder'] = load_class(name, folder)(cfg.decoder)
        self.num_scales = self.networks['decoder'].num_scales

        self.scale_inv_depth = SigmoidToInvDepth(
            min_depth=cfg.min_depth, max_depth=cfg.max_depth)

    # def _get_output_shape(self, module, img_shape, out_shape):
    #     # TODO(soonminh): is it a bad habit?
    #     dummy = torch.zeros((1, 3, *img_shape))
    #     outputs = module(dummy)
    #     scale_and_shapes = []
    #     for out in outputs:
    #         C, H, W = out.shape[1:]
    #         assert int(img_shape[0] / H) == int(img_shape[1] / W)
    #         scale = int(img_shape[0] / H)
    #         ishape = (C, H, W)
    #         oshape = (C, int(out_shape[0]/scale), int(out_shape[1]/scale))
    #         scale_and_shapes.append((scale, ishape, oshape))
    #     return scale_and_shapes

    def forward(self, batch, return_logs=False):
        """Network forward pass"""
        # Prepare meta information for MultiDepthSweepFusion
        # TODO(soonminh): check the ability of decoder to learn depth prediction
        #                   from dynamic multi-camera configuration
        #                   (e.g. from multiple datasets simultaneously)
        decoder_required_keys = ('intrinsics', 'pose_to_pano')
        meta_info = {}
        for cam, sample in batch.items():
            if not cam.startswith('camera'):
                continue
            meta_info[cam] = {k: sample[k] if 'pano' not in cam else sample[k]
                                for k in decoder_required_keys if k in sample}

        log_images = {}

        if self.freeze_encoders:
            with torch.no_grad():
                # Get per-camera features from encoders
                per_camera_features = {key: self.networks['encoders'][key](sample['rgb'])
                                        for key, sample in batch.items() if 'rgb' in sample}
        else:
            # Get per-camera features from encoders
            per_camera_features = {key: self.networks['encoders'][key](sample['rgb'])
                                    for key, sample in batch.items() if 'rgb' in sample}

        # Predict depth from multi-cam features
        out = self.networks['decoder'](per_camera_features, meta_info)

        inv_depths = [out[('output', i)] for i in range(self.num_scales)]
        inv_depths = [self.scale_inv_depth(inv_depth) for inv_depth in inv_depths]

        return {
            'inv_depths': inv_depths,
            **log_images,
        }
