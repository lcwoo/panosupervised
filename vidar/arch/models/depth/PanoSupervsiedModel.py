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

@iterate1
def make_rgb_scales(rgb, pyramid):
    return [interpolate_image(rgb, shape=pyr.shape[-2:]) for pyr in pyramid]


class PanoSupervsiedModel(BaseModel):
    """
    PanoDepth model with Supervision

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self._input_keys = ('rgb', 'intrinsics', 'pose_to_pano', 'rays_embedding')
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
        camera_pano = PanoCamera(K_pano, hw_pano, Twc=Twc_pano, name='camera_pano').scaled(0.5)
        
        depth_gt = resize_torch_preserve(batch_cam['depth'][0][0].detach().cpu(), (128, 1024))
        

        imag_depths, pano_images = [], []
        downsample_ratio, upsample_ratio = 0.5, 2  # Upsample ratio is the inverse of downsample

        # Process each camera in the batch
        for cam_name in filter(lambda k: k.startswith('camera_0'), batch.keys()):
            batch_cam = batch[cam_name]
            K, Tcw = batch_cam['intrinsics'][0], batch_cam['extrinsics'][0].inverse()
            camera = Camera(K, batch_cam['rgb'][0].shape[-2:], Tcw=Tcw.float(), name=cam_name).scaled(0.5 * downsample_ratio)

            # Reconstruct and project depth map
            world_points = camera_pano.reconstruct_depth_map(panodepth[0], to_world=True)
            coords = camera.project_points(world_points, from_world=True, normalize=True)[0]
            imag_depth, weights = self.flow_reverse(panodepth[0], coords, dst_img_shape=[int(s * 0.5) for s in batch_cam['rgb'][0].shape[-2:]])

            # Adjust depth using interpolation
            imag_depth = F.interpolate(imag_depth, scale_factor=upsample_ratio, mode='bilinear', align_corners=True)
            weights = F.interpolate(weights, scale_factor=upsample_ratio, mode='bilinear', align_corners=True)
            imag_depths.append(imag_depth / (weights + 1e-6))

            pano_images.append(F.grid_sample(batch_cam['rgb'][0], coords, mode='bilinear', padding_mode='zeros', align_corners=True))
            
        pano_images = torch.stack(pano_images, axis=1)
        num_views = (pano_images != 0.0).sum(axis=1).clamp(min=1.0)
        pano_image = pano_images.sum(axis=1) / num_views

        return depth_gt, pano_image

    def forward(self, batch, epoch, return_logs=False):
        """Forward pass with supervision to compute and optimize depth estimation."""
        ctx = 0
        filtered_batch = {cam: {k: sample[k][ctx] for k in self._input_keys if k in sample} for cam, sample in batch.items() if is_dict(sample)}
        

        # Compute inverse depth and convert to actual depth
        net_output = self.networks['depth'](filtered_batch, return_logs)
        
        pred_panodepth = inv2depth(net_output['inv_depths'])
        output_dict = {'predictions': {'panodepth': {0: pred_panodepth}}}

        depth_gt, pano_image = self.to_per_camera_depth(batch, pred_panodepth)
        output_dict['predictions']['depth'] = [depth_gt]

        camera_names = list(filter(lambda k: k.startswith('camera_0'), batch.keys()))
        output_dict['batch'] = {
            'depth': {0: torch.stack([batch[cam_name]['depth'][0] for cam_name in camera_names], axis=1)},
            'sensor_name': [[n] for n in camera_names],
            **batch
        }
        output_dict['batch']['camera_pano']['rgb'] = {0: pano_image}

        if not self.training:
            return output_dict
        
        losses = self.compute_losses(pano_image, pred_panodepth, depth_gt)
        return {'loss': losses['loss'], 'metrics': losses['metrics'], 'predictions': {'depth': {0: pred_panodepth}}}

    def compute_losses(self, rgb, depths, gt_depths):
        """Compute supervised and smoothness losses for depth estimation."""
        tgt = 0
        rgbs = make_rgb_scales(rgb, depths)
        rgb_tgt = [rgbs[tgt][i] for i in range(len(rgbs[tgt]))]
        import ipdb; ipdb.set_trace()
        

        supervision_output = self.losses['supervised'](depths[0][0], gt_depths[tgt])
        loss = supervision_output['loss']
        metrics = {**supervision_output['metrics']}

        # import ipdb; ipdb.set_trace()
        # smoothness_output = self.losses['smoothness'](rgb_tgt, depths)
        # loss = supervision_output['loss'] + smoothness_output['loss']
        # metrics = {**supervision_output['metrics'], **smoothness_output['metrics']}

        return {'loss': loss, 'metrics': metrics}