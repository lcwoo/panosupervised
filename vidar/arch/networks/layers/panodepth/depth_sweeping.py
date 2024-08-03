from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from vidar.arch.networks.layers.convs import ConvBlock
from vidar.datasets.PanoCamOuroborosDataset import PANO_CAMERA_NAME
from vidar.geometry.pose import Pose


class FeatTransform(nn.Module):
    """
    Feature transformation module using a given depth hypothesis

    Parameters
    ----------
    camera: string
        Camera name for indexing
    in_scale: float
        Scale of input resolution
    in_shape: Tuple of int
        (channel, height, width) of input feature
    out_scale: float
        Scale of output resolution
    out_shape: Tuple of int
        (channel, height, width) of output feature
    given_depth : float
        Uniform depth hypothesis
    """
    def __init__(self, camera, in_scale, in_shape, out_scale, out_shape, given_depth):
        super().__init__()

        self._in_shape = in_shape
        self._out_shape = out_shape
        self._in_scale = in_scale
        self._out_scale = out_scale

        self._camera_name = camera
        self._given_depth = given_depth

        self.register_buffer('flat_grid', self._precompute_target_grid())

    def __str__(self):
        return f'[{self.__class__.__name__}, depth hypothesis: {self._given_depth}] '\
             + f'camera: {self._camera_name} ({self._in_shape}) -> pano ({self._out_shape}) [scale: {self._scale}]'

    def _precompute_target_grid(self):
        Ho, Wo = self._out_shape[1:]
        ys = torch.linspace(0, Ho - 1, Ho)
        xs = torch.linspace(0, Wo - 1, Wo)
        ys, xs = torch.meshgrid([ys, xs], indexing='ij')
        ones = torch.ones_like(xs)
        grid = torch.stack([xs, ys, ones], dim=0)
        flat_grid = grid.view(-1, 3, Ho * Wo)
        return flat_grid

    def to_polar(self, x, y):
        return (x ** 2 + y ** 2 + 1e-8).sqrt(), torch.atan2(y, x + 1e-8)

    def to_cartesian(self, rho, phi):
        return rho * torch.cos(phi), rho * torch.sin(phi)

    def compute_grid(self, in_intrinsic, out_intrinsic, relative_pose):
        """Compute a fixed mapping from pano space to camera space"""
        B = len(in_intrinsic)
        K_out_inv = out_intrinsic.inverse()
        K_in      = in_intrinsic

        xnorm_out_polar = torch.matmul(K_out_inv, self.flat_grid)

        phi = xnorm_out_polar[:, 0]
        yy = xnorm_out_polar[:, 1]
        rho = torch.ones_like(phi)  # TODO(soonminh): Assuming unit cylinder, rho == 1

        xx, zz = self.to_cartesian(rho, phi)
        xnorm_out_cartesian = torch.stack([xx, yy, zz], dim=1)

        xnorm_out = xnorm_out_cartesian * self._given_depth
        xnorm_in = relative_pose * xnorm_out
        X_cam = torch.matmul(K_in, xnorm_in)

        Ci, Hi, Wi = self._in_shape
        behind_img_plane = X_cam[:, 2] < 1e-5
        Zn = X_cam[:, 2]
        Xn = X_cam[:, 0] / (Zn + 1e-8)
        Yn = X_cam[:, 1] / (Zn + 1e-8)
        Xnorm = 2 * Xn / (Wi - 1) - 1.
        Ynorm = 2 * Yn / (Hi - 1) - 1.

        Xnorm[behind_img_plane] = -10
        Ynorm[behind_img_plane] = -10

        Ho, Wo = self._out_shape[1:]
        ref_coords = torch.stack([Xnorm, Ynorm], dim=-1).view(-1, Ho, Wo, 2)
        return ref_coords

    def forward(self, x, meta):
        in_intrinsic = meta[self._camera_name]['intrinsics'].clone().float()
        out_intrinsic = meta[PANO_CAMERA_NAME]['intrinsics'].clone().float()
        # if self._camera_name == PANO_CAMERA_NAME:
        #     import ipdb; ipdb.set_trace()
        relative_pose = Pose(meta[self._camera_name]['pose_to_pano'].clone().float().inverse())

        scaler = 1.0 / self._in_scale
        in_intrinsic[..., 0, 0] *= scaler
        in_intrinsic[..., 1, 1] *= scaler
        in_intrinsic[..., 0, 2]  = (in_intrinsic[..., 0, 2] + 0.5) * scaler - 0.5
        in_intrinsic[..., 1, 2]  = (in_intrinsic[..., 1, 2] + 0.5) * scaler - 0.5

        scaler = 1.0 / self._out_scale
        out_intrinsic[..., 0, 0] *= scaler
        out_intrinsic[..., 1, 1] *= scaler
        out_intrinsic[..., 0, 2] *= scaler
        out_intrinsic[..., 1, 2] *= scaler

        ref_coords = self.compute_grid(in_intrinsic, out_intrinsic, relative_pose)
        transformed_feat = F.grid_sample(x, ref_coords, padding_mode='zeros', align_corners=True)
        return transformed_feat


class MultiDepthTransform(nn.Module):
    """
    Feature transformation and aggregation module using a set of given depth hypothesis

    Parameters
    ----------
    camera: string
        Camera name for indexing
    in_scale: float
        Scale of input resolution
    in_shape: Tuple of int
        (channel, height, width) of input feature
    out_scale: float
        Scale of output resolution
    out_shape: Tuple of int
        (channel, height, width) of output feature
    given_depths : List of float
        A set of hypothesized depth
    """
    def __init__(self, camera, in_scale, in_shape, out_scale, out_shape, given_depths):
        super().__init__()
        self.transforms = nn.ModuleList(
            [FeatTransform(camera, in_scale, in_shape, out_scale, out_shape, d) for d in given_depths]
        )

    def forward(self, x, meta):
        # Apply transformations for given multiple depth values
        feats_list = [T(x, meta) for T in self.transforms]
        return feats_list
