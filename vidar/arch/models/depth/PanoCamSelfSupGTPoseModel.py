from vidar.arch.models.BaseModel import BaseModel
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
        self._input_keys = ('rgb', 'intrinsics', 'pose_to_pano')

    def forward(self, batch, return_logs=False, **kwargs):
        # Only RGB(t=0) will be forwarded to infer depth
        t = 0
        filtered_batch = {}
        for cam, sample in batch.items():
            if is_dict(sample):
                filtered_batch[cam] = {k: sample[k][t] if 'pano' not in cam else sample[k]
                                            for k in self._input_keys if k in sample}

        ### Compute depth
        # 1. Compute inverse depth
        net_output = self.networks['depth'](filtered_batch)

        # TODO(soonminh): add pose network (require rgb_context)

        loss_dict = self.losses['reprojection'](batch, net_output, return_logs, use_gtpose=self.gt_pose)
        assert 'loss' in loss_dict and 'metrics' in loss_dict

        return {
            'predictions': {
                'panodepth': {0: inv2depth(net_output['inv_depths'][0])},
            },
            **net_output,
            **loss_dict,
        }

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
