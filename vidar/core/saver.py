# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import os

from vidar.datasets.augmentations.resize import resize_torch_preserve
from vidar.utils.config import cfg_has
from vidar.utils.data import make_list
from vidar.utils.types import is_dict, is_list
from vidar.utils.viz import viz_depth, viz_optical_flow
from vidar.utils.write import write_depth, write_image, write_pickle, write_npz


class Saver:
    """
    Wandb logger class to monitor training

    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    ckpt : String
        Name of the model checkpoint (used to create the save folder)
    """
    def __init__(self, cfg, ckpt=None):
        self.folder = cfg_has(cfg, 'folder', None)

        self.rgb = make_list(cfg.rgb) if cfg_has(cfg, 'rgb') else []
        self.depth = make_list(cfg.depth) if cfg_has(cfg, 'depth') else []
        self.pose = make_list(cfg.pose) if cfg_has(cfg, 'pose') else []
        self.optical_flow = make_list(cfg.optical_flow) if cfg_has(cfg, 'optical_flow') else []

        self.store_data = cfg_has(cfg, 'store_data', False)
        self.separate = cfg.has('separate', False)

        self.ckpt = None
        if ckpt is not None:
            if is_list(ckpt):
                self.ckpt = ckpt[0].split('/')[-2]
            else:
                self.ckpt = os.path.splitext(os.path.basename(ckpt))[0]

        self.naming = cfg_has(cfg, 'naming', 'filename')
        assert self.naming in ['filename', 'splitname'], \
            'Invalid naming for saver: {}'.format(self.naming)

    def get_filename(self, path, batch, idx, i, c):
        """Get filename based on input information"""
        if self.naming == 'filename':
            # TODO(soonminh): do we need i here (batch?)
            filename = os.path.join(path, batch['filename'][0][c][i]).replace('{}/', '')
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            return filename
        elif self.naming == 'splitname':
            raise NotImplementedError('Please check filename format.')
            if self.separate:
                return os.path.join(path, '%010d' % idx, '%010d' % idx)
            else:
                return os.path.join(path, '%010d' % idx)
        else:
            raise NotImplementedError('Invalid naming for saver: {}'.format(self.naming))

    def save_data(self, batch, output, prefix):
        """
        Prepare for data saving

        Parameters
        ----------
        batch : Dict
            Dictionary with batch information
        output : Dict
            Dictionary with output information
        prefix : String
            Prefix string for the log name
        """
        if self.folder is None:
            return

        idx = batch['idx']
        predictions = output['predictions']

        path = os.path.join(self.folder, prefix)
        if self.ckpt is not None:
            path = os.path.join(path, self.ckpt)
        os.makedirs(path, exist_ok=True)
        self.save(batch, predictions, path, idx, 0)

    def save(self, batch, predictions, path, idx, i):
        """
        Save batch and prediction information

        Parameters
        ----------
        batch : Dict
            Dictionary with batch information
        predictions : Dict
            Dictionary with output predictions
        path : String
            Path where data will be saved
        idx : Int
            Batch index in the split
        i : Int
            Index within batch

        Returns
        -------
        data : Dict
            Dictionary with output data that was saved
        """
        raw_intrinsics = batch['raw_intrinsics'][0][i].cpu() if 'raw_intrinsics' in batch else \
            batch['intrinsics'][0][i].cpu() if 'intrinsics' in batch else None
        intrinsics = batch['intrinsics'][0][i].cpu() if 'intrinsics' in batch else None

        data = {
            'raw_intrinsics': raw_intrinsics,
            'intrinsics': intrinsics,
        }
        batch = batch['camera_pano']
        img_shape = batch['rgb'][0].shape[-2:]

        for key in batch.keys():

            if key.startswith('rgb'):
                data[key + '_gt'] = {k: v[i].cpu() for k, v in batch[key].items()}
                for ctx in batch[key].keys():
                    rgb = batch[key][ctx][i].cpu()
                    if 'gt' in self.rgb:
                        if rgb.dim() == 4:
                            # TODO(sohwang): Should be improved
                            for j in range(len(rgb)):
                                filename = self.get_filename(path, batch, idx, i, j)
                                # write_image('%s_%s(%d_%d)_gt.png' % (filename, key, j, ctx),
                                write_image('%s_%s_%d_gt.png' % (filename, key, ctx),
                                            rgb[j])
                        else:
                            filename = self.get_filename(path, batch, idx, i, 0)
                            # write_image('%s_%s(%d)_gt.png' % (filename, key, ctx),
                            write_image('%s_%s(%d)_gt.png' % (filename, key, ctx),
                                        rgb)

            if key.startswith('depth'):
                data[key + '_gt'] = {k: v[i].cpu() for k, v in batch[key].items()}
                for ctx in batch[key].keys():
                    depth = batch[key][ctx][i].cpu()
                    if 'gt_png' in self.depth:
                        # write_depth('%s_%s(%d)_gt.png' % (filename, key, ctx),
                        filename = self.get_filename(path, batch, idx, i, 0)
                        write_depth('%s_%s_%d_gt.png' % (filename, key, ctx),
                                    depth)
                    if 'gt_npz' in self.depth:
                        if depth.dim() == 4:
                            for j in range(len(depth)):
                                # write_depth('%s_%s(%d_%d)_gt_viz.png' % (filename, key, j, ctx),
                                filename = self.get_filename(path, batch, idx, i, j)
                                write_depth('%s_%s_%d_gt_viz.png' % (filename, key, ctx),
                                            depth[j], intrinsics=raw_intrinsics)
                        else:
                            # write_depth('%s_%s(%d)_gt.npz' % (filename, key, ctx),
                            filename = self.get_filename(path, batch, idx, i, 0)
                            write_depth('%s_%s_%d_gt.npz' % (filename, key, ctx),
                                        depth, intrinsics=raw_intrinsics)
                    if 'gt_viz' in self.depth:
                        if depth.dim() == 4:
                            for j in range(len(depth)):
                                # write_image('%s_%s(%d_%d)_gt_viz.png' % (filename, key, j, ctx),
                                filename = self.get_filename(path, batch, idx, i, j)
                                write_image('%s_%s_%d_gt_viz.png' % (filename, key, ctx),
                                            viz_depth(resize_torch_preserve(depth[j], img_shape), filter_zeros=True))
                        else:
                            # write_image('%s_%s(%d)_gt_viz.png' % (filename, key, ctx),
                            filename = self.get_filename(path, batch, idx, i, 0)
                            write_image('%s_%s_%d_gt_viz.png' % (filename, key, ctx),
                                        viz_depth(resize_torch_preserve(depth, img_shape), filter_zeros=True))

            if key.startswith('pose'):
                pose = {k: v[i].cpu() for k, v in batch[key].items()}
                data[key + '_gt'] = pose
                if 'gt' in self.pose:
                    filename = self.get_filename(path, batch, idx, i, 0)
                    write_pickle('%s_%s_gt' % (filename, key),
                                 pose)

        for key in predictions.keys():

            if key.startswith('rgb'):
                data[key + '_pred'] = {k: v[i].cpu() for k, v in predictions[key].items()}
                for ctx in predictions[key].keys():
                    rgb = predictions[key][ctx][i].cpu()
                    if 'pred' in self.rgb:
                        if rgb.dim() == 5:
                            for j in range(rgb.shape[1]):
                                # write_image('%s_%s(%d_%d)_pred.png' % (filename, key, j, ctx),
                                filename = self.get_filename(path, batch, idx, i, j)
                                write_image('%s_%s_%d_pred.png' % (filename, key, ctx),
                                            rgb[:, j])
                        else:
                            # write_image('%s_%s(%d)_pred.png' % (filename, key, ctx),
                            filename = self.get_filename(path, batch, idx, i, 0)
                            write_image('%s_%s_%d_pred.png' % (filename, key, ctx),
                                        rgb)

            if key.startswith('panodepth'):
                data[key + '_pred'] = {k: v[i].cpu() for k, v in predictions[key].items()}
                for ctx in predictions[key].keys():
                    depth = predictions[key][ctx][i]

                    if 'png' in self.depth:
                        if depth.dim() == 4:
                            for j in range(len(depth)):
                                # write_depth('%s_%s(%d_%d)_pred.png' % (filename, key, j, ctx),
                                filename = self.get_filename(path, batch, idx, i, j)
                                write_depth('%s_%s_%d_pred.png' % (filename, key, ctx),
                                            depth[j])
                        else:
                            # write_depth('%s_%s(%d)_pred.png' % (filename, key, ctx),
                            filename = self.get_filename(path, batch, idx, i, 0)
                            write_depth('%s_%s_%d_pred.png' % (filename, key, ctx),
                                        depth)
                    if 'npz' in self.depth:
                        if depth.dim() == 4:
                            for j in range(len(depth)):
                                # write_depth('%s_%s(%d_%d)_pred.npz' % (filename, key, j, ctx),
                                filename = self.get_filename(path, batch, idx, i, j)
                                write_depth('%s_%s_%d_pred.npz' % (filename, key, ctx),
                                            depth[j], intrinsics=intrinsics)
                        else:
                            # write_depth('%s_%s(%d)_pred.npz' % (filename, key, ctx),
                            filename = self.get_filename(path, batch, idx, i, 0)
                            write_depth('%s_%s_%d_pred.npz' % (filename, key, ctx),
                                        depth, intrinsics=intrinsics)
                    if 'viz' in self.depth:
                        if depth.dim() == 4:
                            for j in range(len(depth)):
                                # write_image('%s_%s(%d_%d)_pred_viz.png' % (filename, key, j, ctx),
                                filename = self.get_filename(path, batch, idx, i, j)
                                write_image('%s_%s_%d_pred_viz.png' % (filename, key, ctx),
                                            viz_depth(resize_torch_preserve(depth[j], img_shape)))
                        else:
                            # write_image('%s_%s(%d)_pred_viz.png' % (filename, key, ctx),
                            filename = self.get_filename(path, batch, idx, i, 0)
                            write_image('%s_%s_%d_pred_viz.png' % (filename, key, ctx),
                                        viz_depth(resize_torch_preserve(depth, img_shape)))

            if key.startswith('pose'):
                pose = {key: val[i].cpu() for key, val in predictions[key].items()}
                data[key + '_pred'] = pose
                if 'pred' in self.pose:
                    filename = self.get_filename(path, batch, idx, i, 0)
                    write_pickle('%s_%s_pred' % (filename, key),
                                 pose)

            if key.startswith('fwd_optical_flow'):
                optical_flow = {key: val[i].cpu() for key, val in predictions[key].items()}
                data[key + '_pred'] = optical_flow
                if 'npz' in self.optical_flow:
                    filename = self.get_filename(path, batch, idx, i, 0)
                    write_npz('%s_%s_pred' % (filename, key),
                                 {'fwd_optical_flow': optical_flow})
                if 'viz' in self.optical_flow:
                    for ctx in optical_flow.keys():
                        # write_image('%s_%s(%d)_pred_viz.png' % (filename, key, ctx),
                        filename = self.get_filename(path, batch, idx, i, 0)
                        write_image('%s_%s_%d_pred_viz.png' % (filename, key, ctx),
                                    viz_optical_flow(optical_flow[ctx]))

            if key.startswith('mask'):
                if is_dict(predictions[key]):
                    data[key] = {k: v[i].cpu() for k, v in predictions[key].items()}
                    for ctx in data[key].keys():
                        # write_image('%s_%s(%d)_pred_viz.png' % (filename, key, ctx), predictions[key][ctx][0])
                        filename = self.get_filename(path, batch, idx, i, 0)
                        write_image('%s_%s_%d_pred_viz.png' % (filename, key, ctx), predictions[key][ctx][0])
                elif is_list(predictions[key]):
                    data[key] = [v[i].cpu() for k, v in predictions[key]]
                    for ctx in data[key]:
                        # write_image('%s_%s(%d)_pred_viz.png' % (filename, key, ctx), predictions[key][ctx][0])
                        filename = self.get_filename(path, batch, idx, i, 0)
                        write_image('%s_%s_%d_pred_viz.png' % (filename, key, ctx), predictions[key][ctx][0])
                else:
                    data[key] = predictions[key][i].cpu()
                    filename = self.get_filename(path, batch, idx, i, 0)
                    write_image('%s_%s_pred_viz.png' % (filename, key), predictions[key][0])

        if self.store_data:
            write_pickle('%s' % filename, data)

        return data

class PanoSaver(Saver):
    def __init__(self, cfg, ckpt=None):
        super().__init__(cfg, ckpt)
        assert 'gt' not in self.rgb, 'GT RGB is not defined in PanoSpace.'

    def get_filename(self, path, batch_cam, batch_idx, i):
        """Get filename based on input information"""
        if self.naming == 'filename':
            # TODO(soonminh): do we need i here (batch?)
            filename = os.path.join(path, batch_cam['filename'][0][i]).replace('{}/', '')
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            return filename
        elif self.naming == 'splitname':
            return os.path.join(path, '%010d' % batch_idx)
        else:
            raise NotImplementedError('Invalid naming for saver: {}'.format(self.naming))

    def save(self, batch, predictions, path, idx, i):
        """
        Save batch and prediction information

        Parameters
        ----------
        batch : Dict
            Dictionary with batch information
        predictions : Dict
            Dictionary with output predictions
        path : String
            Path where data will be saved
        idx : Int
            Batch index in the split
        i : Int
            Index within batch

        Returns
        -------
        data : Dict
            Dictionary with output data that was saved
        """
        ctx = 0
        intrinsics = {c: s['intrinsics'][ctx].cpu().numpy()
                        for c, s in batch.items() if c.startswith('camera')}
        data = {
            'intrinsics': intrinsics,
        }
        ## Save per-camera GTs (including camera_pano)
        for cam, batch_cam in batch.items():
            if not cam.startswith('camera'):
                continue

            if 'rgb' in batch_cam:
                for key in self.rgb:
                    rgb = batch_cam['rgb'][ctx][i].cpu()
                    filename = self.get_filename(path, batch_cam, idx, i)
                    write_image(f'{filename}_rgb_{key}.png', rgb)
                    data[f'rgb_{cam}'] = rgb

            if 'depth' in batch_cam:
                depth = batch_cam['depth'][ctx][i].cpu()
                data[f'depth_gt_{cam}'] = depth
                filename = self.get_filename(path, batch_cam, idx, i)

                for key in self.depth:
                    if key == 'gt_png':
                        write_depth(f'{filename}_depth_{key}.png',
                            depth[0])
                    elif key == 'gt_npz':
                        write_depth(f'{filename}_depth_{key}.npz',
                            depth, intrinsics=intrinsics[cam])
                    elif key == 'gt_viz':
                        img_shape = [s.item() for s in batch_cam['hw']] if 'hw' in batch_cam else batch_cam['rgb'][ctx].shape[-2:]
                        write_depth(f'{filename}_depth_{key}.png',
                            viz_depth(resize_torch_preserve(depth, img_shape), filter_zeros=True))

            # TODO(soonminh): Save pose/mask if needed

        ## Save Preds: panodepth
        cam = 'camera_pano'
        batch_cam = batch[cam]
        depth_pred = predictions['panodepth'][ctx][i][0].cpu()
        data[f'depth_pr_camera_pano'] = depth_pred
        filename = self.get_filename(path, batch_cam, idx, i)

        # TODO(soonminh): Save pano-rgb if needed

        for key in self.depth:
            if 'gt' in key:
                continue
            if key == 'png':
                write_depth(f'{filename}_depth_{key}.png',
                    depth_pred[0])
            elif key == 'npz':
                write_depth(f'{filename}_depth_{key}.npz',
                    depth_pred, intrinsics=intrinsics[cam])
            elif key == 'viz':
                img_shape = [s.item() for s in batch_cam['hw']] if 'hw' in batch_cam else batch_cam['rgb'][ctx].shape[-2:]
                write_depth(f'{filename}_depth_{key}.png',
                    viz_depth(resize_torch_preserve(depth_pred, img_shape), filter_zeros=True))

        if self.store_data:
            write_pickle('%s' % filename.replace(cam.upper(), ''), data)

        return data
