import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.cache = None

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=0)
        return torch.flatten(emb, 0, 1)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cache is not None and self.cache.shape == tensor.shape:
            return self.cache

        self.cache = None
        # batch_size, x, y, orig_ch = tensor.shape
        batch_size, orig_ch, y, x = tensor.shape
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_y = torch.einsum("i,j->ij", self.inv_freq, pos_y)
        sin_inp_x = torch.einsum("i,j->ij", self.inv_freq, pos_x)
        emb_y = self.get_emb(sin_inp_y).unsqueeze(2).repeat(1, 1, x)
        emb_x = self.get_emb(sin_inp_x).unsqueeze(1).repeat(1, y, 1)

        emb = torch.cat([emb_y, emb_x], axis=0)[None]
        self.cache = emb.repeat(tensor.shape[0], 1, 1, 1)
        return self.cache


class MultiDepthSweepFusion(nn.Module):
    """
    Cylindrical depth sweeping based fusion module using multiple depth hypotheses
    Project features for all views (per-camera + per-depth), apply view-attention, then apply a ResNet block

    Parameters
    ----------
    scale_and_shapes: Dict of {String: Tuple(in_scale, in_shape, out_scale, out_shape)}
        input and output shapes of this module per camera
    view_attention: bool
        turn on/off view attention
    positional_encoding: int
        the number of channels for positional encoding
    depth_hypotheses: List of float
        Hypothesized scalar depth values
    """
    def __init__(self, scale_and_shapes, view_attention=False, positional_encoding=0, depth_hypotheses=[3, 5, 10, 30]):
        super().__init__()
        self.per_camera_transforms = nn.ModuleDict({
            camera: MultiDepthTransform(camera, *shapes, given_depths=depth_hypotheses)
                for camera, shapes in scale_and_shapes.items()})

        # TODO(soonminh): Assume all camera features have the same shape
        _, in_shape, _, out_shape = list(scale_and_shapes.values())[0]
        num_cameras = len(scale_and_shapes.keys())
        num_depths = len(depth_hypotheses)

        self.prepare_att = None
        if view_attention:
            self.prepare_att = nn.Conv2d(in_shape[0] * num_cameras * num_depths, num_cameras * num_depths, kernel_size=3, padding=1)
        self.conv = BasicBlock(in_shape[0], out_shape[0], downsample=None)

        self.get_pos_enc = None
        if positional_encoding > 0:
            self.get_pos_enc = PositionalEncoding2D(positional_encoding)


    def forward(self, feats, meta, return_logs=False):
        # 1. Cylinderical sweeping for all views (per-camera + per-depth)
        panofeats = []
        for cam, feat in feats.items():
            panofeat = self.per_camera_transforms[cam](feat, meta)
            panofeats.extend(panofeat)

        # 2. Aggregate multicam cylindrical features
        num_views = torch.concat([panofeat.detach().sum(axis=1, keepdim=True) != 0.0 for panofeat in panofeats], axis=1)
        num_views = num_views.sum(axis=1, keepdim=True).clamp(min=1.0)

        if self.prepare_att is None:
            out = torch.stack(panofeats, axis=1).sum(axis=1) / num_views
        else:
            multicam_panofeat = torch.stack(panofeats, axis=1)

            # 3. View-attention
            # Get {num_cams x num_depths}-dimensional attention mask and expand it (by broadcasting)
            # TODO(soonminh): Check if .detach() is required for attn
            attn = torch.sigmoid(self.prepare_att(torch.cat(panofeats, axis=1))).unsqueeze(2)
            multicam_panofeat = (multicam_panofeat * attn).sum(axis=1) / num_views
            out = self.conv(multicam_panofeat)

        # 3. TODO(soonminh): cylindrical padding to improve boundaries

        # 4. Positional encoding
        if self.get_pos_enc is not None:
            pos_enc = self.get_pos_enc(out)
            out = torch.cat([out, pos_enc], axis=1)

        return out
