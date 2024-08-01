import math
from copy import deepcopy
import torch.nn.functional as F

import torch

from vidar.geometry.camera import Camera
from vidar.geometry.pose import Pose
from vidar.utils.tensor import pixel_grid, norm_pixel_grid, cat_channel_ones

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
    def __init__(self, K, hw, Twc=None, Tcw=None, name=''):
        super().__init__(K, hw, Twc, Tcw, name)

        self.rho = 1.0

    def to_polar(self, z, x):
        return (x ** 2 + z ** 2 + 1e-8).sqrt(), torch.atan2(z, x + 1e-8)

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
        fy =  camera_cfg['height'] / (z_max - z_min)
        ty =   camera_cfg['height'] * (1 + z_min / (z_max - z_min))

        hw = [camera_cfg['height'], camera_cfg['width']]
        # TODO: (chungwoo) K하고 Twc를 gpu에 올리수 있도록 수정 )
        
        K = torch.FloatTensor([
            [fx,  0, tx],
            [ 0, fy, ty],
            [ 0,  0,  1]])
        
        # TODO(chungwoo): Compute rotation from config
        # Twc[ 1, 1] *= -1 # to make phi clockwise

        # Twc = torch.eye(4, dtype=torch.float32)
        # Twc[ 1, 1] *= -1    # to make phi clockwise
        # Twc[:3, 3] = - torch.FloatTensor(camera_cfg['position_in_world'])
        return {'K': K, 'hw': hw}
    
    #TODO: make mask wrap function for pano camera
    def project_mask_to_pano(mask, Twc_i, K_i):
        # Step 1: Create a meshgrid for the mask
        height, width = mask.shape[1], mask.shape[2]
        grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        grid_z = torch.ones_like(grid_x)  # Assuming a depth of 1; adjust if depth information is available
        
        # Flatten the grid to (N, 3) where N is number of points in the mask
        grid_mask = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1).float()

        # Step 2: Transform grid points to camera space using inverse intrinsic matrix K^-1
        # Assume homogenous coordinates for the grid points
        grid_points_hom = torch.cat([grid_mask[:, :2], torch.ones_like(grid_x).flatten().unsqueeze(-1)], dim=1)
        inv_K = torch.inverse(K_i)
        points_camera = (inv_K @ grid_points_hom.T).T

        # Multiply z coordinate to scale the x, y coordinates in the camera space
        points_camera[:, :2] *= grid_mask[:, 2:3]
        points_camera_hom = torch.cat([points_camera, torch.ones(points_camera.shape[0], 1)], dim=1)  # Make it [N, 4]
        points_camera_hom_transposed = points_camera_hom.permute(1, 0)  # Make it [4, N]

        inv_Twc_i = torch.inverse(Twc_i)
        points_world_hom = inv_Twc_i @ points_camera_hom_transposed

        # Optionally, handle the mask data transformation
        # For binary masks, this step might be simply copying the mask. For more complex masks, additional steps might be needed.
        mask_values = mask.flatten()

        # Step 4: Process transformed points and mask values for further usage
        # This is where you would normally project these points onto a panorama or use them in your pipeline.

        return points_world_hom, mask_values
    
    def project_mask(Twc, K, hw, points_world_hom, mask_values):
        """
        Project world points and corresponding mask values onto the image plane.

        Parameters
        ----------
        points_world_hom : torch.Tensor
            Input 3D world points in homogeneous coordinates [N, 4]
        mask_values : torch.Tensor
            Corresponding mask values [N]
        K : torch.Tensor
            Camera intrinsic matrix [3, 3]

        Returns
        -------
        projected_mask : torch.Tensor
            Projected 2D mask with values from the input mask
        """
        # Assuming Twc (camera extrinsic matrix) is already part of the class
        # Convert world coordinates to camera coordinates
        
        if points_world_hom.dim() == 3 and points_world_hom.shape[0] == 1:
            points_world_hom = points_world_hom.squeeze(0)
            
        # Convert world coordinates to camera coordinates
        points_camera = (Twc @ points_world_hom).T

        # Drop the homogeneous coordinate
        points_camera = points_camera[:, :3]

        # Project points using the camera intrinsic matrix
        points_2d = (K @ points_camera.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]  # Normalize by the depth component

        # Convert to pixel coordinates (assuming the output dimensions are known, e.g., height and width)
        height, width = hw  # Ensure these are set according to your image dimensions
        pixels_x = torch.clamp((points_2d[:, 0] + 1) * width / 2, 0, width - 1).long()
        pixels_y = torch.clamp((points_2d[:, 1] + 1) * height / 2, 0, height - 1).long()
        # Create the projected mask
        projected_mask = torch.zeros((height, width), device=points_world_hom.device)
        valid_indices = (pixels_x >= 0) & (pixels_x < width) & (pixels_y >= 0) & (pixels_y < height)
        import ipdb; ipdb.set_trace()
        projected_mask[pixels_y[valid_indices], pixels_x[valid_indices]] = mask_values[valid_indices]

        return projected_mask

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
        xnorm_polar = torch.matmul(self.invK[:, :3, :3].detach().to(depth.device), grid)

        phi, yy = xnorm_polar[:, 0] , xnorm_polar[:, 1]
        xx, zz = self.to_cartesian(self.rho, phi)
        xnorm = torch.stack([xx, yy, zz], dim=1).view(b, 3, -1)

        # Scale rays to metric depth
        points = xnorm * depth.view(b, 1, -1)

        if to_world and self.Tcw is not None:
            points = self.Tcw.detach().to(depth.device) * points

        # import ipdb; ipdb.set_trace()
        return points.view(b, 3, -1)

    def reconstruct_cost_volume(self, volume, to_world=True, flatten=True):
        raise NotImplementedError

    def Pwc(self, from_world=True):
        raise NotImplementedError
    
    def project_points_with_cam1(self, points, Twc, from_world=True, normalize=True, return_z=False):
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
        device = points.device

        if is_depth_map:
            points = points.reshape(points.shape[0], 3, -1)
        b, _, n = points.shape

        # points = torch.matmul(self.Pwc(from_world), cat_channel_ones(points, 1))
        if from_world:
            Xc = torch.matmul(Twc.to(device), cat_channel_ones(points, 1))
        else:
            Xc = points

        # Cartesian -> Polar
        Xp_rho, Xp_pi = self.to_polar(Xc[:, 2], Xc[:,0])
        Xp_z = Xc[:, 1] / Xp_rho * self.rho

        # Project 3D points onto the camera image plane
        points = self.K.to(device).bmm(
            torch.stack([Xp_pi, Xp_z, torch.ones_like(Xp_pi, device=device), torch.ones_like(Xp_pi, device=device)], axis=1))


        coords = points[:, :2] / (points[:, 2].unsqueeze(1) + 1e-8)
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
        device = points.device

        if is_depth_map:
            points = points.reshape(points.shape[0], 3, -1)
        b, _, n = points.shape

        # points = torch.matmul(self.Pwc(from_world), cat_channel_ones(points, 1))

        if from_world:
            Xc = self.Twc.to(device) * points
        else:
            Xc = points

        # Cartesian -> Polar
        Xp_rho, Xp_pi = self.to_polar(Xc[:, 2], Xc[:, 0])
        Xp_z = Xc[:, 1] / Xp_rho * self.rho

        # Project 3D points onto the camera image plane
        points = self.K.to(device).bmm(
            torch.stack([Xp_pi, Xp_z, torch.ones_like(Xp_pi, device=device), torch.ones_like(Xp_pi, device=device)], axis=1))


        coords = points[:, :2] / (points[:, 2].unsqueeze(1) + 1e-8)
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