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


class PanoCamSelfSupGTPoseModel(BaseModel):
    """
    PanoDepth model

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__(cfg)

        self.gt_pose = True
        self._input_keys = ('rgb', 'intrinsics', 'pose_to_pano', 'rays_embedding')
        self.produce_to_per_camera_depth = cfg_has(cfg.model, 'produce_to_per_camera_depth', False)
        self.produce_pano_image = cfg_has(cfg.model, 'produce_pano_image', False)
        self.flow_reverse = FlowReversal()

    def to_per_camera_depth(self, batch, panodepth):
        K_pano = batch[PANO_CAMERA_NAME]['intrinsics'][0].float()
        Twc_pano = batch[PANO_CAMERA_NAME]['Twc'].float()
        hw_pano = batch[PANO_CAMERA_NAME]['depth'][0].shape[-2:]
        camera_pano = PanoCamera(K_pano, hw_pano, Twc=Twc_pano, name='camera_pano')

        scale_factor = 0.5
        camera_pano = camera_pano.scaled(scale_factor)

        downsample_ratio = 0.5
        upsample_ratio = int(1 / downsample_ratio)


        imag_depths = []
        pano_images = []

        camera_names = [k for k in batch.keys() if k.startswith('camera_0')]
        for cam_name in camera_names:
            batch_cam = batch[cam_name]

            hw = batch_cam['rgb'][0].shape[-2:]
            K = batch_cam['intrinsics'][0]     # Intrinsics are not changed over time
            Tcw = batch_cam['extrinsics'][0].inverse()
            camera = Camera(K, hw, Tcw=Tcw.float(), name=cam_name)

            scaled_imag_shape = [int(num * scale_factor * downsample_ratio) for num in hw]

            camera = camera.scaled(scale_factor * downsample_ratio)


            world_points = camera_pano.reconstruct_depth_map(panodepth[0], to_world=True)
            coords = camera.project_points(world_points, from_world=True, normalize=True)[0]
            imag_depth, weights = self.flow_reverse(panodepth[0], coords, dst_img_shape=scaled_imag_shape)

            imag_depth = F.interpolate(imag_depth, scale_factor=upsample_ratio, mode='bilinear', align_corners=True)
            weights = F.interpolate(weights, scale_factor=upsample_ratio, mode='bilinear', align_corners=True)

            imag_depths.append(imag_depth / (weights + 1e-6))

            if self.produce_pano_image:
                # TODO(soonminh): need to use valid to filter our some area?
                warped = F.grid_sample(batch_cam['rgb'][0], coords,
                        mode='bilinear', padding_mode='zeros', align_corners=True)
                pano_images.append(warped)

            # import pdb; pdb.set_trace()
            # from PIL import Image
            # import numpy as np
            # from vidar.utils.viz import flow_to_color, viz_depth
            # Image.fromarray((flow_to_color(coords[0].detach().cpu().numpy()) * 255.0).astype(np.uint8)).save('flow_pano.png')
            # Image.fromarray((viz_depth(panodepth[0]) * 255.0).astype(np.uint8)).save('tmp_pano.png')
            # Image.fromarray((viz_depth(imag_depths[-1]) * 255.0).astype(np.uint8)).save('tmp.png')

        if self.produce_pano_image:
            pano_images = torch.stack(pano_images, axis=1)
            num_views = (pano_images != 0.0).sum(axis=1).clamp(min=1.0)
            pano_image = pano_images.sum(axis=1) / num_views
        else:
            pano_image = None

        return torch.stack(imag_depths, axis=1), pano_image

    # def to_pano_image(self, batch, panodepth):


    def forward(self, batch, return_logs=False, **kwargs):
        # Only RGB(ctx=0) will be forwarded to infer depth
        ctx = 0
        filtered_batch = {}
        for cam, sample in batch.items():
            if is_dict(sample):
                filtered_batch[cam] = {k: sample[k][ctx] for k in self._input_keys if k in sample}

        ### Compute depth
        log_images = {}
        import ipdb; ipdb.set_trace()

        # 1. Compute inverse depth
        net_output = self.networks['depth'](filtered_batch, return_logs)
        log_images.update(net_output.pop('log_images'))

        panodepth = inv2depth(net_output['inv_depths'])

        output_dict = {
            'predictions': {
                'panodepth': {0: panodepth},
            },
        }

        if self.produce_to_per_camera_depth and not self.training:
            # TODO(soonminh): Convert per-camera depth to compare with others if nedded
            depth, pano_image = self.to_per_camera_depth(batch, panodepth)
            output_dict['predictions']['depth'] = [depth]

            # HACK(soonminh): for per-camera evaluation
            camera_names = [k for k in batch.keys() if k.startswith('camera_0')]
            output_dict['batch'] = {
                'depth': {
                    0: torch.stack([batch[cam_name]['depth'][0] for cam_name in camera_names], axis=1),
                },
                'sensor_name': [[n] for n in camera_names],
                **batch,
            }
            if pano_image is not None:
                output_dict['batch']['camera_pano']['rgb'] = {0: pano_image}

        # TODO(soonminh): remove this and plot validation losses if needed
        if not return_logs and not self.training:
            return output_dict

        # TODO(soonminh): add pose network (require rgb_context)

        loss_dict = self.losses['reprojection'](batch, net_output, return_logs, use_gtpose=self.gt_pose)
        assert 'loss' in loss_dict and 'metrics' in loss_dict
        log_images.update(loss_dict.pop('log_images'))

        output_dict['log_images'] = log_images
        output_dict.update(**net_output)
        output_dict.update(**loss_dict)
        return output_dict

        #########################################################################################################
        # ## DEBUG
        # from vidar.utils.write import write_image
        # from vidar.utils.viz import viz_depth, viz_inv_depth
        # import cv2
        # import numpy as np
        # camera_order = ['camera_07', 'camera_05', 'camera_01', 'camera_06', 'camera_08', 'camera_09']
        # float_tensor_to_uint8_numpy = lambda x: (x[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        # scene, *_, timestamp = batch['camera_pano']['filename'][0].split('/')
        # filename = f'{scene}_{timestamp}'

        # gt_pano_depth = (viz_depth(batch['camera_pano']['depth'][0][0]) * 255.0).astype(np.uint8)
        # h_viz, w_viz = gt_pano_depth.shape[:2]

        # pr_pano_depth = (viz_inv_depth(inv_depths[0]) * 255.0).astype(np.uint8)
        # pr_pano_depth = cv2.resize(pr_pano_depth, (w_viz, h_viz))

        # raw_rgb = [float_tensor_to_uint8_numpy(batch[c]['raw_rgb'][0]) for c in camera_order]
        # raw_rgb = np.hstack(raw_rgb)
        # _h, _w = raw_rgb.shape[:2]
        # raw_rgb = cv2.resize(raw_rgb, (0, 0), fx=w_viz/_w, fy=w_viz/_w)
        # cv2.imwrite(f'{filename}_raw_rgb.png', raw_rgb[:,:,(2,1,0)])

        # aug_rgb = [float_tensor_to_uint8_numpy(batch[c]['rgb'][0]) for c in camera_order]
        # aug_rgb = np.hstack(aug_rgb)
        # _h, _w = aug_rgb.shape[:2]
        # aug_rgb = cv2.resize(aug_rgb, (0, 0), fx=w_viz/_w, fy=w_viz/_w)
        # cv2.imwrite(f'{filename}_aug_rgb.png', aug_rgb[:,:,(2,1,0)])
        # cv2.imwrite(f'{filename}_frame.png', np.vstack([raw_rgb, aug_rgb, gt_pano_depth, pr_pano_depth])[:,:,(2,1,0)])
        #########################################################################################################
