# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

import torch
import torch.nn as nn

from vidar.arch.blocks.depth.SigmoidToInvDepth import SigmoidToInvDepth
from vidar.arch.networks.BaseNet import BaseNet
from vidar.utils.config import cfg_has, get_folder_name, load_class

_KNOWN_DECODER_TYPES = (
    'PanoDepthDecoder',             # Proposed
    'SurroundDepthDecoder',         # SurroundDepth
)

def allow_multicam_input(func):
    """Decorator to allow 5-dim input tensors by reshaping, e.g. BxNxCxHxW -> {BxN}xCxHxW"""
    # HACK(soonminh): this decorator works only for below network.
    # TODO(soonminh): should be improved and moved to vidar/utils/decorator.py
    def inner(cls, batch, **kwargs):
        B = N = None
        rgb = batch['rgb']
        if rgb.dim() == 5:
            B, N = rgb.shape[:2]
            rgb = rgb.view(B * N, *rgb.shape[2:])
            batch['rgb'] = rgb

        out = func(cls, batch, **kwargs)
        if B is not None:
            # HACK(soonminh): Assuming a specific output type; a dict of list of tensor
            out = {k: [v.view(B, N, *v.shape[1:]) for v in vs] for k, vs in out.items()}
        return out
    return inner


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

        ### Encoders
        if cfg_has(cfg, 'share_encoder', False):
            # SurroundDepth implementation
            file = cfg_has(cfg.encoder, 'file', 'encoders/ResNetEncoder')
            folder, name = get_folder_name(file, 'networks')
            encoder_module = load_class(name, folder)(cfg.encoder)
            self.networks['encoder'] = encoder_module
            cfg.decoder.num_ch_enc = encoder_module.num_ch_enc
        else:
            # PanoDepth implementation
            self.input_cameras = [c for c in cfg.encoders.keys() if c.startswith('camera')]
            self.freeze_encoders = cfg_has(cfg, 'freeze_encoders', False)

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

    @allow_multicam_input
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

        if 'encoders' in self.networks:
            # Per-camera encoders, for PanoDepth
            if self.freeze_encoders:
                with torch.no_grad():
                    # Get per-camera features from encoders
                    per_camera_features = {key: self.networks['encoders'][key](sample['rgb'])
                                            for key, sample in batch.items() if 'rgb' in sample}
            else:
                # Get per-camera features from encoders
                per_camera_features = {key: self.networks['encoders'][key](sample['rgb'])
                                        for key, sample in batch.items() if 'rgb' in sample}
        else:
            # Shared encoder
            per_camera_features = self.networks['encoder'](batch['rgb'])

        # Predict depth from multi-cam features
        out = self.networks['decoder'](per_camera_features, meta_info)

        inv_depths = [out[('output', i)] if ('output', i) in out else out[('disp', i)]
                        for i in range(self.num_scales)]
        inv_depths = [self.scale_inv_depth(inv_depth) for inv_depth in inv_depths]

        return {
            'inv_depths': inv_depths,
            **log_images,
        }
