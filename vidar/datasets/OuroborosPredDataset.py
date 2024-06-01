import cv2
import numpy as np
import os
from PIL import Image

import torch

from vidar.datasets.OuroborosDataset import OuroborosDataset
from vidar.utils.logging import pcolor

class OuroborosPredDataset(OuroborosDataset):
    def __init__(self, save_path, **kwargs):
        super().__init__(**kwargs)
        self.save_path = save_path

        font = {'color': 'blue', 'attrs': ('bold', 'dark')}
        print(pcolor('#' + ' PREDICTION PATH: {} '.format(save_path), **font))

    def __getitem__(self, idx):
        samples = super().__getitem__(idx)

        pred_depths_viz = {}
        pred_depths_npz = {}
        raw_shape = (samples['depth'][0].shape[3], samples['depth'][0].shape[2])
        for context in range(-self.bwd_context, self.fwd_context + 1):

            pred_depth_viz = []
            pred_depth_npz = []
            for c in range(self.num_cameras):
                scene, _, camera, ts = self.get_filename(idx, c, context).split('/')
                pred_depth_viz_np = np.array(Image.open(
                    os.path.join(self._path, scene, camera, '{}_depth_{}_pred_viz.png'.format(ts, 0))
                ))
                pred_depth_viz_np = cv2.resize(pred_depth_viz_np, raw_shape, cv2.INTER_CUBIC)
                pred_depth_viz.append(pred_depth_viz_np)

                pred_depth_npz_np = np.load(
                    os.path.join(self.save_path, scene, camera, '{}_depth_{}_pred.npz'.format(ts, 0))
                )['depth']
                pred_depth_npz_np = cv2.resize(pred_depth_npz_np, raw_shape, cv2.INTER_CUBIC)
                pred_depth_npz.append(pred_depth_npz_np)

            pred_depth_viz = np.stack(pred_depth_viz, axis=0).astype(np.float32) / 255.0
            pred_depths_viz[context] = pred_depth_viz

            pred_depth_npz = np.stack(pred_depth_npz, axis=0)[:, None]
            pred_depths_npz[context] = torch.from_numpy(pred_depth_npz)


            # Conver to torch.tensor to enable "flip"
            # TODO(soonminh): need more work on display/display_sample.py:L115
            # IndexError: index 3 is out of bounds for dimension 0 with size 1
            # pred_depths[context] = torch.from_numpy(np.stack(pred_depth, axis=0).astype(np.float32) / 255.0)

        # for context in range(1, self.fwd_context + 1):
        #     pred_depth = []
        #     for c in range(self.num_cameras):
        #         scene, _, camera, ts = self.get_filename(idx, c, context).split('/')
        #         pred_depth_np = np.array(Image.open(
        #             os.path.join(self.save_path, scene, camera, '{}_depth_{}_pred_viz.png'.format(ts, 0))
        #         ))
        #         pred_depth.append(pred_depth_np.transpose((2, 0, 1)))
        #     pred_depths[context] = np.stack(pred_depth, axis=0)
        samples['pred_depth_viz'] = pred_depths_viz
        samples['pred_depth_npz'] = pred_depths_npz
        return samples
