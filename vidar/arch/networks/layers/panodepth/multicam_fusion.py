import math
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

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
    agg_op: str
        Aggregation method
    """
    def __init__(self, scale_and_shapes, depth_hypotheses=[3, 5, 10, 30], agg_op='concat'):
        super().__init__()
        self.per_camera_transforms = nn.ModuleDict({
            camera: MultiDepthTransform(camera, *shapes, given_depths=depth_hypotheses)
                for camera, shapes in scale_and_shapes.items()})

        # TODO(soonminh): Assume all camera features have the same shape
        _, in_shape, _, out_shape = list(scale_and_shapes.values())[0]

        assert agg_op in ('concat'), 'Unknown aggregation operation: {}'.format(agg_op)
        self.agg_op = agg_op

        if agg_op == 'concat':
            # Order of aggregation: given depths -> camera (previous)
            # Order of aggregation: camera -> given depths (WIP)
            self.conv = BasicBlock(in_shape[0], out_shape[0], downsample=None)
        elif agg_op == 'self_attention':
            # TODO(soonminh): Implement attention-based fusion
            raise NotImplementedError

    def forward(self, feats, meta, return_logs=False):
        """
        feats: dictionary of features
            e.g., feats = {camera: feat for cam, feat in feats.items()}
        """
        # 1. Per-camera / Per-depth cylindrical sweeping
        panofeats = []
        for cam, feat in feats.items():
            panofeat = self.per_camera_transforms[cam](feat, meta)
            panofeats.extend(panofeat)

        # 2. Aggregate multicam cylindrical features
        if self.agg_op == 'concat':
            num_views = torch.concat([panofeat.detach().sum(axis=1, keepdim=True) != 0.0 for panofeat in panofeats], axis=1)
            num_views = num_views.sum(axis=1, keepdim=True).clamp(min=1.0)
            multicam_panofeat = torch.stack(panofeats, axis=1).sum(axis=1) / num_views
        else:
            raise NotImplementedError

        # 3. TODO(sohwang): cylindrical padding to improve boundaries

        out = self.conv(multicam_panofeat)

        return out
