import numpy as np
import torch
import torch.nn.functional as F

from vidar.arch.networks.layers.panodepth.flow_reversal import FlowReversal
from vidar.arch.models.BaseModel import BaseModel
from vidar.datasets.PanoCamOuroborosDataset import PANO_CAMERA_NAME
from vidar.geometry.camera import Camera
from vidar.geometry.camera_pano import PanoCamera
from vidar.utils.config import cfg_has
from vidar.utils.types import is_dict
from vidar.utils.depth import inv2depth
from vidar.utils.decorators import iterate1
from vidar.utils.tensor import interpolate_image
from vidar.datasets.augmentations.resize import resize_npy_preserve, resize_torch_preserve
from vidar.utils.viz import viz_depth

@iterate1
def make_rgb_scales(rgb, pyramid):
    return [interpolate_image(rgb, shape=pyr.shape[-2:]) for pyr in pyramid]


class PanoSupervisedModel(BaseModel):
    """
    PanoDepth model with Supervision

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self._input_keys = ('rgb', 'intrinsics', 'pose_to_pano', 'rays_embedding','depth','angle')
        self.produce_to_per_camera_depth = cfg_has(cfg.model, 'produce_to_per_camera_depth', False)
        self.produce_pano_image = cfg_has(cfg.model, 'produce_pano_image', False)
        self.flow_reverse = FlowReversal()

    def to_per_camera_depth(self, batch, panodepth):
        """Convert panoramic depth map to per-camera depth maps."""
        # Extract and scale the panoramic camera configuration
        batch_cam = batch['camera_pano']
        K_pano = batch_cam['intrinsics'][0].float()
        Twc_pano = batch_cam['Twc'].float()
        hw_pano = batch_cam['depth'][0].shape[-2:]
        panodepth = batch_cam['depth']
        camera_pano = PanoCamera(K_pano, hw_pano, Twc=Twc_pano, name='camera_pano').scaled(0.5)
        xyz_lidar = camera_pano.reconstruct_depth_map(panodepth[0], to_world=False)
        xyz_lidar = xyz_lidar.view(3, -1)
        np_xyz_lidar = xyz_lidar.detach().cpu().numpy()
        rgb_lidar = np.zeros_like(np_xyz_lidar).T

        for cam_name in filter(lambda k: k.startswith('camera_0'), batch.keys()):
            data = batch[cam_name]
            K = batch_cam['intrinsics'][0]
            Tcw = data['extrinsics'][0].inverse()
            Tpc = data['pose_to_pano'][0].inverse()
            import ipdb; ipdb.set_trace()
            
            # Transform the points to the current camera coordinate system
            ones = torch.ones(1, xyz_lidar.shape[1], device=xyz_lidar.device)
            xyz_lidar_hom = torch.cat([xyz_lidar, ones], dim=0) 
            xyz_camera = Tpc[:3, :3] @ xyz_lidar + Tpc[:3, 3:].unsqueeze(1)
            iz = xyz_camera[2, :]
            ix = (xyz_camera[0, :] / iz).long()
            iy = (xyz_camera[1, :] / iz).long()

            # Define the camera resolution
            cam_width, cam_height = 1024, 128  # Example dimensions, replace with actual dimensions if variable

            # Check if points are within the image bounds
            valid_ix = (ix >= 0) & (ix < cam_width)
            valid_iy = (iy >= 0) & (iy < cam_height)
            valid_iz = (iz > 0)
            valid_indices = valid_ix & valid_iy & valid_iz

            # Get RGB values from image and assign to rgb_lidar
            image = data['rgb'][0].permute(1, 2, 0).contiguous()  # Make sure it's contiguous for memory layout
            colors = image[iy[valid_indices], ix[valid_indices]]
            rgb_lidar[valid_indices] = colors.type(torch.uint8)
            

        return depth_gt, pano_image

    def forward(self, batch, epoch, return_logs=False):
        """Forward pass with supervision to compute and optimize depth estimation."""
        ctx = 0
        filtered_batch = {cam: {k: sample[k][ctx] for k in self._input_keys if k in sample} for cam, sample in batch.items() if is_dict(sample)}
        

        # Compute inverse depth and convert to actual depth
        net_output = self.networks['depth'](filtered_batch, return_logs)
        
        pred_panodepth = inv2depth(net_output['inv_depths'])
        
        gt_panodepth = batch['camera_pano']['depth'][0]
        pred_panodepth_lastlayer = resize_torch_preserve(pred_panodepth[0], (256, 2048))
        output_dict = {'predictions': {'panodepth': {0: pred_panodepth}}}
        output_dict = {
            'predictions': {
                'panodepth': {0: pred_panodepth},
            },
            'gt_panodepth': {
                'gt_panodepth': {0: gt_panodepth}
            }
        }
                
        if not self.training:
            return output_dict
        
        losses = self.compute_losses(pred_panodepth_lastlayer, gt_panodepth)
        return {'loss': losses['loss'], 'metrics': losses['metrics'], 'predictions': {'depth': {0: pred_panodepth},'gt_panodepth': {"gt_panodepth":{0: gt_panodepth}}}}

    def compute_losses(self, depths, gt_depths):
        """Compute supervised and smoothness losses for depth estimation."""
        
        supervision_output = self.losses['supervised'](depths, gt_depths)
        loss = supervision_output['loss']
        metrics = {**supervision_output['metrics']}
        
        return {'loss': loss, 'metrics': metrics}