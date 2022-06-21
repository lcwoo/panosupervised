import math
from copy import deepcopy

import torch

from vidar.geometry.camera import Camera
from vidar.geometry.pose import Pose
from vidar.utils.tensor import pixel_grid, norm_pixel_grid

class PanoCamera(Camera):
    """
    Differentiable Panoramic Camera class for 3D reconstruction

    Parameters
    ----------
    K : torch.Tensor
        Camera intrinsics [B,3,3]
    hw : Tuple
        Camera height and width
    Twc : Pose or torch.Tensor
        Camera pose (world to camera) [B,4,4]
    Tcw : Pose or torch.Tensor
        Camera pose (camera to world) [B,4,4]
    """
    def __init__(self, K, hw, Twc=None, Tcw=None):
        super().__init__(K, hw, Twc, Tcw)

        # TODO(sohwang): read rho from config
        self.rho = 1.0

    def to_polar(self, x, y):
        return (x ** 2 + y ** 2).sqrt(), torch.atan2(y, x)

    def to_cartesian(self, rho, phi):
        return rho * torch.cos(phi), rho * torch.sin(phi)

    @staticmethod
    def params_from_config(camera_cfg):
        """
        Calculate intrinsic/extrinsinc camera parameters from config
            Note:
                - PanoCamera coordinate: Forward/Left/Up (x-axis, y-axis, z-axis)
        """
        MUST_HAVE_KEYS = ('width', 'height', 'rho', 'phi_range', 'z_range', 'position_in_world')
        assert all([k in camera_cfg for k in MUST_HAVE_KEYS]), \
            'Incomplete camera config. Need {}, but only has {}.'.format(
                MUST_HAVE_KEYS, camera_cfg.keys())

        phi_min, phi_max = camera_cfg['phi_range']
        fx = camera_cfg['width'] / (phi_max - phi_min)
        tx = camera_cfg['width'] / (phi_max - phi_min) * (math.pi - phi_min)

        z_min, z_max = camera_cfg['z_range']
        fy = - camera_cfg['height'] / (z_max - z_min)
        ty =   camera_cfg['height'] * (1 + z_min / (z_max - z_min))

        hw = (camera_cfg['height'], camera_cfg['width'])
        K = torch.FloatTensor([
            [fx,  0, tx],
            [ 0, fy, ty],
            [ 0,  0,  1]])

        # TODO(sohwang): Compute rotation from config
        Twc = torch.eye(4, dtype=torch.float32)
        Twc[ 1, 1] *= -1    # to make phi clockwise
        Twc[:3, 3] = - torch.FloatTensor(camera_cfg['position_in_world'])
        return {'K': K, 'hw': hw, 'Twc': Twc}

    def reconstruct_depth_map(self, depth, to_world=False):
        """
        Reconstruct a depth map from the camera viewpoint

        Parameters
        ----------
        depth : torch.Tensor
            Input depth map [B,1,H,W]
        to_world : Bool
            Transform points to world coordinates

        Returns
        -------
        points : torch.Tensor
            Output 3D points [B,3,H,W]
        """
        if depth is None:
            return None

        b, _, h, w = depth.shape
        grid = pixel_grid(depth, with_ones=True, device=depth.device).view(b, 3, -1)
        xnorm_polar = torch.matmul(self.invK[:, :3, :3], grid)

        phi, zz = xnorm_polar[:, 0], xnorm_polar[:, 1]
        # rho = self.rho * torch.ones_like(phi)

        xx, yy = self.to_cartesian(self.rho, phi)
        xnorm = torch.stack([xx, yy, zz], dim=1).view(b, 3, -1)

        # Scale rays to metric depth
        points = xnorm * depth.view(b, 1, -1)

        if to_world and self.Tcw is not None:
            points = self.Tcw * points
        return points.view(b, 3, h, w)

    def reconstruct_cost_volume(self, volume, to_world=True, flatten=True):
        raise NotImplementedError

    def Pwc(self, from_world=True):
        raise NotImplementedError

    def project_points(self, points, from_world=True, normalize=True, return_z=False):
        """
        Project points back to image plane

        Parameters
        ----------
        points : torch.Tensor
            Input 3D points [B,3,H,W] or [B,3,N]
        from_world : Bool
            Whether points are in the global frame
        normalize : Bool
            Whether projections should be normalized to [-1,1]
        return_z : Bool
            Whether projected depth is return as well

        Returns
        -------
        coords : torch.Tensor
            Projected 2D coordinates [B,2,H,W]
        depth : torch.Tensor
            Projected depth [B,1,H,W]
        """
        is_depth_map = points.dim() == 4
        hw = self._hw if not is_depth_map else points.shape[-2:]

        if is_depth_map:
            points = points.reshape(points.shape[0], 3, -1)
        b, _, n = points.shape

        # points = torch.matmul(self.Pwc(from_world), cat_channel_ones(points, 1))

        if from_world:
            Xc = self.Twc * points
        else:
            Xc = points

        # Cartesian -> Polar
        Xp_rho, Xp_pi = self.to_polar(Xc[:, 0], Xc[:, 1])
        Xp_z = Xc[:, 2] / Xp_rho * self.rho

        # Project 3D points onto the camera image plane
        points = self.K.bmm(
            torch.stack([Xp_pi, Xp_z, torch.ones_like(Xp_pi), torch.ones_like(Xp_pi)], axis=1))


        coords = points[:, :2] / (points[:, 2].unsqueeze(1) + 1e-7)
        # depth = points[:, 2]
        depth = Xp_rho

        if not is_depth_map:
            if normalize:
                coords = norm_pixel_grid(coords, hw=self._hw, in_place=True)
                invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                          (coords[:, 1] < -1) | (coords[:, 1] > 1) | (depth < 0)
                coords[invalid.unsqueeze(1).repeat(1, 2, 1)] = -2
            if return_z:
                return coords.permute(0, 2, 1), depth
            else:
                return coords.permute(0, 2, 1)

        coords = coords.view(b, 2, *hw)
        depth = depth.view(b, 1, *hw)

        if normalize:
            coords = norm_pixel_grid(coords, hw=self._hw, in_place=True)
            invalid = (coords[:, 0] < -1) | (coords[:, 0] > 1) | \
                      (coords[:, 1] < -1) | (coords[:, 1] > 1) | (depth[:, 0] < 0)
            coords[invalid.unsqueeze(1).repeat(1, 2, 1, 1)] = -2

        if return_z:
            return coords.permute(0, 2, 3, 1), depth
        else:
            return coords.permute(0, 2, 3, 1)

    def project_cost_volume(self, points, from_world=True, normalize=True):
        raise NotImplementedError