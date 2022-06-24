import torch
import torch.nn as nn

from vidar.arch.networks.layers.panodepth.depth_sweeping import MultiDepthTransform


class MultiDepthSweepFusion(nn.Module):
    """
    Cylindrical depth sweeping based fusion module using multiple depth hypotheses

    Parameters
    ----------
    scale_and_shapes: Dict of {String: Tuple(scale, in_shape, out_shape)}
        input and output shapes of this module per camera
    depth_hypotheses: List of float
        Hypothesized scalar depth values
    """
    def __init__(self, scale_and_shapes, depth_hypotheses=[3, 5, 10, 30]):
        super().__init__()
        self.per_camera_transforms = nn.ModuleDict({
            camera: MultiDepthTransform(camera, *shapes, given_depths=depth_hypotheses)
                for camera, shapes in scale_and_shapes.items()})

    def forward(self, feats, meta):
        # 1. Per-camera multi-depth cylindrical sweeping + self-attention
        panofeats = []
        for cam, feat in feats.items():
            panofeat = self.per_camera_transforms[cam](feat, meta)
            panofeats.append(panofeat)

        # 2. Aggregate multicam cylindrical features
        # TODO(sohwang): better way to compute num_views
        num_views = torch.concat([panofeat.detach().sum(axis=1, keepdim=True) != 0.0 for panofeat in panofeats], axis=1)
        num_views = num_views.sum(axis=1, keepdim=True).clamp(min=1.0)
        multicam_panofeat = torch.stack(panofeats, axis=1).sum(axis=1) / num_views

        # 3. TODO(sohwang): cylindrical padding to improve boundaries

        return multicam_panofeat
