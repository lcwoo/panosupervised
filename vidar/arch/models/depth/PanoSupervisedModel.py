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

        self._input_keys = ('rgb', 'intrinsics', 'pose_to_pano', 'depth', 'mask')
        self.produce_to_per_camera_depth = cfg_has(cfg.model, 'produce_to_per_camera_depth', False)
        self.produce_pano_image = cfg_has(cfg.model, 'produce_pano_image', False)
        self.flow_reverse = FlowReversal()

    # def create_pano_mask(self, batch, panodepth):
    #     camera_names = [k for k in batch.keys() if k.startswith('camera_0')]
    #     pano_masks = []
    #     K_pano = batch[PANO_CAMERA_NAME]['intrinsics'][0].float()
    #     Twc_pano = batch[PANO_CAMERA_NAME]['Twc'].float()
    #     hw_pano = batch[PANO_CAMERA_NAME]['depth'][0].shape[-2:]
    #     camera_pano = PanoCamera(K_pano, hw_pano, Twc=Twc_pano, name='camera_pano')
    #     warp_masks=[]
    #     for cam_name in camera_names:
    #         batch_cam = batch[cam_name]
    #         mask = batch_cam['mask']  # Assuming mask shape is [1, H, W]

    #         # Camera setup (you need to ensure the Camera class has the methods used here)
    #         hw = mask.shape[-2:]
    #         K = batch_cam['intrinsics'][0].float()
    #         Tcw = batch_cam['extrinsics'][0].inverse().float()
    #         camera = Camera(K, hw, Tcw=Tcw, name=cam_name)

    #         # Convert mask to panorama coordinates
    #         warp_mask = camera.reconstruct_depth_map(mask, to_world=True)
    #         panomask = camera_pano.project_points(warp_mask, from_world=True, normalize=True)[0]
    #         warp_masks.append(warp_mask)
    #     warp_masks = torch.cat(warp_masks, dim=0)
    #     pano_mask = camera_pano.project_points(warp_masks, from_world=True, normalize=True)[0]

    #     if pano_masks:
    #         stacked_masks = torch.stack(pano_masks, dim=0)
    #         pano_mask = stacked_masks.max(dim=0)[0]  # Using max to combine overlapping masks
    #     else:
    #         # If no masks are available, return a dummy or error
    #         raise ValueError("No camera masks available to create a panorama mask.")

    #     return pano_mask

    
    def rgb_lidar(self, batch, panodepth):
        K_pano = batch[PANO_CAMERA_NAME]['intrinsics'][0].float()
        Twc_pano = batch[PANO_CAMERA_NAME]['Twc'].float()
        hw_pano = batch[PANO_CAMERA_NAME]['depth'][0].shape[-2:]
        camera_pano = PanoCamera(K_pano, hw_pano, Twc=Twc_pano, name='camera_pano')
        scale_factor = 0.5
        camera_pano = camera_pano.scaled(scale_factor)

        downsample_ratio = 0.5
        upsample_ratio = int(1 / downsample_ratio)

        imag_depths = []
        sampled_depths =[]
        pano_images = []
        camera_names = [k for k in batch.keys() if k.startswith('camera_0')]
        world_points = camera_pano.reconstruct_depth_map(panodepth[0], to_world=False)
        for cam_name in camera_names:
            batch_cam = batch[cam_name]
            Tcw_i = batch['camera_01']['extrinsics'][0].inverse()
            
            hw = batch_cam['rgb'][0].shape[-2:]
            K = batch_cam['intrinsics'][0]     # Intrinsics are not changed over time
            Tcw = batch_cam['extrinsics'][0].inverse()
            camera = Camera(K, hw, Tcw=Tcw.float(), name=cam_name)

            scaled_imag_shape = [int(num * scale_factor * downsample_ratio) for num in hw]

            camera = camera.scaled(scale_factor * downsample_ratio)
            #I want to use flow reversal to make the gt dense, and the overlay the RGB data forem each camera onto it
            
            coords = camera.project_points(world_points, from_world=True, normalize=True)[0]
            imag_depth, weights = self.flow_reverse(panodepth[0], coords, dst_img_shape=scaled_imag_shape)

            imag_depth = F.interpolate(imag_depth, scale_factor=upsample_ratio, mode='bilinear', align_corners=True)
            weights = F.interpolate(weights, scale_factor=upsample_ratio, mode='bilinear', align_corners=True)
            
            sampled_depth = F.grid_sample((imag_depth / (weights + 1e-6)), coords, mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled_depths.append(sampled_depth)
        
        
        imag_depths = torch.stack(sampled_depths, axis=1)
        num_views = (imag_depths != 0.0).sum(axis=1).clamp(min=1.0)
        pano_reverse_depth = imag_depths.sum(axis=1) / num_views
        # from PIL import Image
        # Image.fromarray((viz_depth(pano_reverse_depth[0]) * 255.0).astype(np.uint8)).save('pano_reverse_depth.png')
        batch_size, _, H, W = pano_reverse_depth.shape
        world_reverse_points = camera_pano.reconstruct_depth_map(pano_reverse_depth, to_world=True).view(batch_size,3, -1)
        rgb_lidar = torch.zeros_like(world_reverse_points, device=world_reverse_points.device).permute(0, 2, 1)

        for cam_name in camera_names:
            batch_cam = batch[cam_name]
            hw = batch_cam['rgb'][0].shape[-2:]
            K = batch_cam['intrinsics'][0]
            Tpc = batch['camera_01']['extrinsics'][0]
            for b in range(batch_size):
                rotation_matrix = Tpc[b, :3, :3]
                translation = Tpc[b, :3, 3].unsqueeze(1)
                
                transformed_points = torch.matmul(rotation_matrix, world_reverse_points[b])
                translation_expanded = translation.repeat(1, transformed_points.shape[1])
                xyz_camera = transformed_points + translation_expanded
                
                xyz = torch.matmul(K[b], xyz_camera)
                ix, iy, iz = xyz[0, :], xyz[1, :], xyz[2, :]
                ix, iy = (ix / iz).long(), (iy / iz).long()
                
                valid_indices = (xyz_camera[2] > 0) & (ix >= 0) & (ix < hw[1]) & (iy >= 0) & (iy < hw[0])
                import ipdb; ipdb.set_trace()
                rgb = batch_cam['rgb'][0][b].permute(1, 2, 0)
                
                rgb_lidar[b][valid_indices] = rgb[ix[valid_indices], iy[valid_indices], :]
                


        import ipdb; ipdb.set_trace()            
        # from PIL import Image
        # pano_tensor = pano_image[0]
        # pano_tensor = pano_tensor.permute(1, 2, 0)
        # pano_tensor = (pano_tensor - pano_tensor.min()) / (pano_tensor.max() - pano_tensor.min())
        # pano_tensor = pano_tensor * 255
        # pano_image_array = pano_tensor.to(torch.uint8)
        # image = Image.fromarray(pano_image_array.cpu().numpy())
        # image.save('pano_reverse_image.png')

        return rgb_lidar
    
    def forward(self, batch, epoch, return_logs=False):
        """Forward pass with supervision to compute and optimize depth estimation."""
        ctx = 0
        filtered_batch = {cam: {k: sample[k][ctx] for k in self._input_keys if k in sample} for cam, sample in batch.items() if is_dict(sample)}

        # Compute inverse depth and convert to actual depth
        net_output = self.networks['depth'](filtered_batch, return_logs)
        # import ipdb; ipdb.set_trace()
        pred_panodepth = inv2depth(net_output['inv_depths'])
        pred_panodepth = resize_torch_preserve(pred_panodepth, (256, 2048))
        gt_panodepth = batch['camera_pano']['depth'][0]
        
        # output_dict = {'predictions': {'panodepth': {0: pred_panodepth}}}
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
        
        
        # TODO:(chungwoo) Implement mask creation
        # pano_mask = self.create_pano_mask(batch,  gt_panodepth.unsqueeze(0))

        # camera_names = [k for k in batch.keys() if k.startswith('camera_0')]
        # output_dict['batch'] = {
        #     'depth': {
        #         0: torch.stack([batch[cam_name]['depth'][0] for cam_name in camera_names], axis=1),
        #     },
        #     'sensor_name': [[n] for n in camera_names],
        #     **batch,
        # }
        # output_dict['batch']['camera_pano']['rgb'] = {0: pano_image}
        
        losses = self.compute_losses( pred_panodepth, gt_panodepth)

        output_dict.update({
            'loss': losses['loss'],
            'metrics': losses['metrics'],
            'predictions': {'depth': {0: pred_panodepth}, 'gt_panodepth': {0: gt_panodepth}}
        })
        
        return output_dict

    def compute_losses(self, depths, gt_depths):
        """Compute supervised and smoothness losses for depth estimation."""
        
        supervision_output = self.losses['supervised'](depths, gt_depths)
          
        loss = supervision_output['loss'] 
        
        metrics = {
            **supervision_output['metrics']
        }
        
        return {'loss': loss, 'metrics': metrics}