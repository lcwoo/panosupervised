import math
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from vidar.arch.networks.layers.panodepth.depth_sweeping import MultiDepthTransform


# https://github.com/tatp22/multidim-positional-encoding
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(math.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x).unsqueeze(1)
        emb_y = self.get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


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
