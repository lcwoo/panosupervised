# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from abc import ABC

import numpy as np
import torch
import torch.nn as nn

from vidar.arch.blocks.depth.SigmoidToInvDepth import SigmoidToInvDepth
from vidar.arch.blocks.depth.SigmoidToDepth import SigmoidToDepth
from vidar.arch.blocks.depth.SigmoidToLogDepth import SigmoidToLogDepth
from vidar.arch.networks.BaseNet import BaseNet
from vidar.utils.config import cfg_has, get_folder_name, load_class
from vidar.utils.tensor import make_same_resolution

_KNOWN_DECODER_TYPES = (
    'PanoDepthDecoder',             # Proposed
    'SurroundDepthDecoder',         # SurroundDepth
)

def allow_multicam_input(func):
    """Decorator to allow 5-dim input tensors by reshaping, e.g. BxNxCxHxW -> {BxN}xCxHxW"""
    # HACK(soonminh): this decorator works only for below network.
    # TODO(soonminh): should be improved and moved to vidar/utils/decorator.py
    def inner(cls, batch, *kargs, **kwargs):
        B = N = None
        rgb = batch.get('rgb', None)
        if rgb is not None and rgb.dim() == 5:
            B, N = rgb.shape[:2]
            rgb = rgb.view(B * N, *rgb.shape[2:])
            batch['rgb'] = rgb

        out = func(cls, batch, *kargs, **kwargs)
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
        if cfg_has(cfg, 'encoders', False):
            # Per-camera encoder
            self.input_cameras = [c for c in cfg.encoders.keys() if c.startswith('camera')]

            scale_and_shapes = dict()
            d_shape = cfg_has(cfg.decoder, 'ref_shape', (256, 2048))
            out_shape = cfg_has(cfg.decoder, 'out_shape', (256, 2048))
            assert d_shape[0] // out_shape[0] == d_shape[1] // out_shape[1], 'Changing aspect ratio is not tested yet.'
            out_scale = d_shape[0] // out_shape[0]

            self.networks['encoders'] = nn.ModuleDict()
            for camera in self.input_cameras:
                cfg_per_cam = cfg.encoders.dict[camera]
                file = cfg_has(cfg_per_cam, 'file', 'MobileNetEncoder')
                folder, name = get_folder_name(file, 'networks')
                encoder_module = load_class(name, folder)(cfg_per_cam)
                self.networks['encoders'][camera] = encoder_module

                e_shape = cfg_has(cfg_per_cam, 'ref_shape', (384, 640))
                scale_and_shapes[camera] = [
                    (in_scale, (ch, e_shape[0]//in_scale, e_shape[1]//in_scale),
                     in_scale * out_scale, (ch, out_shape[0]//in_scale, out_shape[1]//in_scale))
                        for in_scale, ch in zip(encoder_module.reduction, encoder_module.num_ch_enc)]

            cfg.decoder.scale_and_shapes = scale_and_shapes
        else:
            self.input_cameras = cfg.input_cameras

            file = cfg_has(cfg.encoder, 'file', 'encoders/ResNetEncoder')
            folder, name = get_folder_name(file, 'networks')
            encoder_module = load_class(name, folder)(cfg.encoder)
            self.networks['encoder'] = encoder_module
            cfg.decoder.num_ch_enc = encoder_module.num_ch_enc

            # HACK(soonminh)
            if 'PanoDepthDecoder' in cfg.decoder.file:
                e_shape = cfg_has(cfg_per_cam, 'ref_shape', (384, 640))

                d_shape = cfg_has(cfg.decoder, 'ref_shape', (256, 2048))
                out_shape = cfg_has(cfg.decoder, 'out_shape', (256, 2048))
                assert d_shape[0] // out_shape[0] == d_shape[1] // out_shape[1], 'Changing aspect ratio is not tested yet.'
                out_scale = d_shape[0] // out_shape[0]

                scale_and_shapes = dict()
                for camera in self.input_cameras:
                    scale_and_shapes[camera] = [
                    (in_scale, (ch, e_shape[0]//in_scale, e_shape[1]//in_scale),
                     in_scale * out_scale, (ch, out_shape[0]//in_scale, out_shape[1]//in_scale))
                        for in_scale, ch in zip(encoder_module.reduction, encoder_module.num_ch_enc)]

                cfg.decoder.scale_and_shapes = scale_and_shapes

        ### Decoder
        cfg.decoder.input_cameras = self.input_cameras
        folder, name = get_folder_name(cfg.decoder.file, 'networks')
        assert name in _KNOWN_DECODER_TYPES, f'Unknown decoder type: {name}'
        self.networks['decoder'] = load_class(name, folder)(cfg.decoder)
        self.num_scales = self.networks['decoder'].num_scales
        self.depth_focal = cfg_has(cfg.decoder, 'depth_focal', None)

        if cfg.scale_invdepth == 'linear':
            self.scale_inv_depth = SigmoidToDepth(
                min_depth=cfg.min_depth, max_depth=cfg.max_depth)
        elif cfg.scale_invdepth == 'inverse':
            self.scale_inv_depth = SigmoidToInvDepth(
                min_depth=cfg.min_depth, max_depth=cfg.max_depth)
        elif cfg.scale_invdepth == 'exp':
            self.scale_inv_depth = SigmoidToLogDepth()
        else:
            raise NotImplementedError

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
            # Per-camera encoders
            per_camera_features = {key: self.networks['encoders'][key](sample['rgb'])
                                    for key, sample in batch.items() if 'rgb' in sample}
        else:
            # Shared encoder
            if 'rgb' in batch:
                # SurroundDepth
                per_camera_features = self.networks['encoder'](batch['rgb'])
            else:
                per_camera_features = {key: self.networks['encoder'](sample['rgb'])
                                        for key, sample in batch.items() if 'rgb' in sample}

        # Predict depth from multi-cam features
        out = self.networks['decoder'](per_camera_features, meta_info, return_logs)
        log_images.update(out['log_images'])

        inv_depths = [out[('output', i)] if ('output', i) in out else out[('disp', i)]
                        for i in range(self.num_scales)]
        inv_depths = [self.scale_inv_depth(inv_depth) for inv_depth in inv_depths]

        # Concatenate log images
        log_images = {key: np.vstack(make_same_resolution(images, images[0].shape[:2]))
                            for key, images in log_images.items()}

        return {
            'inv_depths': inv_depths,
            'log_images': log_images,
        }
