# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F

from vidar.arch.losses.MultiCamPhotometricLoss import MultiCamPhotometricLoss
from vidar.arch.losses.MultiViewPhotometricLoss import calc_smoothness
from vidar.arch.networks.layers.panodepth.flow_reversal import FlowReversal
from vidar.datasets.PanoCamOuroborosDataset import PANO_CAMERA_NAME
from vidar.geometry.camera import Camera
from vidar.geometry.camera_pano import PanoCamera
from vidar.utils.config import cfg_has
from vidar.utils.depth import inv2depth, depth2inv
from vidar.utils.distributed import print0
from vidar.utils.tensor import match_scales, make_same_resolution
from vidar.utils.viz import viz_photo
from vidar.utils.write import viz_depth

class PanoDepthPhotometricLoss(MultiCamPhotometricLoss):
    def __init__(self, cfg):
        super().__init__(cfg)

        ### Definition of photometric reprojection loss pairs: list of tuples
        # tuple: (target, target_context_idx, source, source_context_idx)
        #   - context_idx: -1 (backward), 0 (current), 1 (forward)
        #   - From "panodepth" prediction, we use flow reversal layer to reconstruct target_depth.
        #     then, use the target_depth to calculate reprojection from source
        loss_pairs = defaultdict(list)
        for tgt, tgt_context_idx, src_cam, src_context_idx in cfg.reprojection_pairs:
            assert (tgt, tgt_context_idx) != (src_cam, src_context_idx), \
                f'No reason to use exactly same coordinate to compute photometric loss: \n' \
                + f'(tgt, tgt_context_idx) (src_cam, src_context_idx): ' \
                + f'({tgt}, {tgt_context_idx}) ({src_cam}, {src_context_idx})'
            loss_pairs[(tgt, tgt_context_idx)].append((src_cam, src_context_idx))
        self.pairs = loss_pairs

        self.mono_weight = cfg_has(cfg, 'mono_weight', 0.9)
        self.stereo_weight = cfg_has(cfg, 'stereo_weight', 0.1)

        self.flow_reverse = FlowReversal()
        self.log_images = defaultdict(list)

    def get_context_and_pose(self, batch_cam, context_idx):
        """Pose: from (ref_cam, context_idx) to (world, 0)"""
        raw_rgb_key = 'raw_rgb' if self.training else 'rgb'
        context = batch_cam[raw_rgb_key][context_idx]
        if context_idx == 0:
            # Cam to world (t=0)
            pose = batch_cam['extrinsics'][0].inverse()
        else:
            # Pose from t=context to t=0, then cam to world (t=0)
            pose = batch_cam['extrinsics'][0].inverse() @ batch_cam['pose'][context_idx].inverse()

        return context, pose

    def get_camera(self, batch, cam_name, context_idx, Tcw):
        K = batch[cam_name]['intrinsics'][0]     # Intrinsics are not changed over time
        hw = batch[cam_name]['rgb'][context_idx].shape[-2:]
        camera = Camera(K, hw, Tcw=Tcw.float(), name=cam_name)
        return camera

    def compute_image_depth(self, pano_camera, imag_camera, pano_depths, return_logs=False):
        """
        Compute depth on camera from pano_depth prediction using flow reversal layer
        """
        pano_width = pano_camera.hw[1]

        downsample_ratio = 0.5
        upsample_ratio = int(1 / downsample_ratio)

        imag_depths = []
        for i in range(self.n):
            DW = pano_depths[i].shape[3]
            scale_factor = DW / float(pano_width)
            # scaled_imag_shape = [int(num * scale_factor * downsample_ratio) for num in imag_camera.hw]
            scaled_imag_shape = [int(num * scale_factor) for num in imag_camera.hw]

            # HACK(soonminh): get downsampled imag_depth from flow_reverse, then upsample it to save computation
            camobj_pano = pano_camera.scaled(scale_factor)
            # camobj_imag = imag_camera.scaled(scale_factor * downsample_ratio)
            camobj_imag = imag_camera.scaled(scale_factor)

            world_points = camobj_pano.reconstruct_depth_map(pano_depths[i], to_world=True)
            coords, valid = camobj_imag.project_points(world_points, from_world=True, normalize=True, return_valid=True)

            # This causes center-dot artifact
            # coords[..., 0] = coords[..., 0] * valid[..., 0].float()
            # coords[..., 1] = coords[..., 1] * valid[..., 0].float()

            imag_depth, weights = self.flow_reverse(pano_depths[i], coords, dst_img_shape=scaled_imag_shape)
            # imag_depth = F.interpolate(imag_depth, scale_factor=upsample_ratio, mode='bicubic', align_corners=True)
            # weights = F.interpolate(weights, scale_factor=upsample_ratio, mode='bicubic', align_corners=True)
            imag_depths.append(imag_depth / (weights + 1e-6))

            if return_logs and i == 0:
                self.log_images['warped_{}'.format(imag_camera.name)].append(
                    (viz_depth(imag_depths[-1][0, 0].detach().cpu()) * 255.0).astype(np.uint8))

        return imag_depths

    def warp(self, camera_tgt, depths_tgt, camera_src, contexts_src):
        width_tgt = camera_tgt.hw[1]
        contexts_warped = []
        for i in range(self.n):
            DW = depths_tgt[i].shape[3]
            scale_factor = DW / float(width_tgt)

            camera_tgt_scaled = camera_tgt.scaled(scale_factor)
            camera_src_scaled = camera_src.scaled(scale_factor)

            world_points = camera_tgt_scaled.reconstruct_depth_map(depths_tgt[i], to_world=True)
            coords, valid = camera_src_scaled.project_points(world_points, from_world=True, normalize=True, return_valid=True)

            # TODO(soonminh): need to use valid to filter our some area?
            warped = F.grid_sample(contexts_src[i], coords,
                        mode='bilinear', padding_mode='zeros', align_corners=self.align_corners)
            contexts_warped.append(warped)
        return contexts_warped

    def reduce_photometric_loss(self, photometric_losses, debug=False):
        """
        Combine the photometric loss from all context images

        Parameters
        ----------
        photometric_losses : list of torch.Tensor [B,3,H,W]
            Pixel-wise photometric losses from the entire context
        debug : boolean
            If set, return per-pixel loss for debugging

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Reduced photometric loss
        per_pixel_photometric_loss : torch.Tensor [B,H,W]
            Per-pixel photometric loss
        """
        # Reduce function
        def view_reduce_func(losses):
            stacked_losses = torch.cat(losses, 1)
            if self.photometric_reduce_op == 'mean':
                return stacked_losses.mean(1, True)
            elif self.photometric_reduce_op == 'min':
                # we need to ignore 'zero' loss because that means no-supervision!
                # HACK(sohwang): better way to compute non-zero mean while keeping all dimensions?
                zero_loss = stacked_losses == 0
                stacked_losses[zero_loss] = torch.finfo(stacked_losses.dtype).max
                min_stacked_losses = stacked_losses.min(1, True)[0]
                min_stacked_losses[zero_loss.min(1, True)[0].bool()] = 0
                return min_stacked_losses
            else:
                raise NotImplementedError(
                    'Unknown photometric_reduce_op: {}'.format(self.photometric_reduce_op))

        def spatial_reduce_func(losses):
            # TODO(sohwang): allow different reduce_op across spatial dimensions
            # TODO(sohwang): to keep original implementation, support 'mean' only for now.
            return losses.mean()

        def scale_reduce_func(losses):
            return sum(losses) / self.n

        # Reduce photometric loss
        view_reduced_loss = [view_reduce_func(photometric_losses[i]) for i in range(self.n)]
        spatial_reduced_loss = [spatial_reduce_func(view_reduced_loss[i]) for i in range(self.n)]
        photometric_loss = scale_reduce_func(spatial_reduced_loss)

        # Return reduced photometric loss
        return (photometric_loss, None) if not debug else (photometric_loss, view_reduced_loss)

    def calc_smoothness_loss_masked(self, inv_depths, images, masks):
        """
        Calculates the smoothness loss for inverse depth maps.

        Parameters
        ----------
        inv_depths : list[torch.Tensor]
            Predicted inverse depth maps for all scales [B,1,H,W]
        images : list[torch.Tensor]
            Original images for all scales [B,3,H,W]

        Returns
        -------
        smoothness_loss : torch.Tensor
            Smoothness loss [1]
        """
        # Calculate smoothness gradients
        smoothness_x, smoothness_y = calc_smoothness(inv_depths, images, self.n)
        # Calculate smoothness loss
        smoothness_loss = sum([(
            (smoothness_x[i] * masks[i][..., :,   :-1].float()).abs().mean() +
            (smoothness_y[i] * masks[i][..., :-1, :].float()).abs().mean()) / 2 ** i
            for i in range(self.n)]) / self.n
        # Apply smoothness loss weight
        smoothness_loss = self.smooth_loss_weight * smoothness_loss
        # Store and return smoothness loss
        return smoothness_loss


    def forward(self, batch, output, return_logs=False, use_gtpose=True):
        pano_depths = inv2depth(output['inv_depths'])
        if return_logs:
            print0(f'(pano_depth) min: {pano_depths[0].min()} / max: {pano_depths[0].max()}')
            self.log_images.clear()
            self.log_images['panodepth'].append(
                (viz_depth(pano_depths[0][0, 0].detach().cpu()) * 255.0).astype(np.uint8))
            self.log_images['panodepth'].append(
                (viz_depth(batch['camera_pano']['depth'][0, 0].detach().cpu()) * 255.0).astype(np.uint8))

        ### Prepare target/source contexts
        K_pano = batch[PANO_CAMERA_NAME]['intrinsics'].float()
        Twc_pano = batch[PANO_CAMERA_NAME]['Twc'].float()
        # TODO(soonminh): Fix this
        # hw_pano = [n.tolist() for n in batch[PANO_CAMERA_NAME]['hw']]
        hw_pano = batch[PANO_CAMERA_NAME]['depth'].shape[-2:]
        camera_pano = PanoCamera(K_pano, hw_pano, Twc=Twc_pano, name='camera_pano')

        losses = {}
        for (cam_tgt, context_idx_tgt), pairs in self.pairs.items():
            context_tgt, T_tgt_to_world = self.get_context_and_pose(batch[cam_tgt], context_idx_tgt)
            camera_tgt = self.get_camera(batch, cam_tgt, context_idx_tgt, T_tgt_to_world)

            if return_logs:
                self.log_images['warped_{}'.format(cam_tgt)].append(
                    (context_tgt[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8))

            # Apply flow reversal to get depth_tgt from depth_pano
            depths_tgt = self.compute_image_depth(camera_pano, camera_tgt, pano_depths, return_logs)
            contexts_tgt = match_scales(context_tgt, depths_tgt, self.n, align_corners=self.align_corners)

            mask_tgt = batch[cam_tgt]['mask']
            masks_tgt = match_scales(mask_tgt, depths_tgt, self.n, align_corners=self.align_corners)

            if return_logs:
                self.log_images['warped_{}'.format(cam_tgt)].append(
                    (mask_tgt[0].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8))

            photometric_losses = [[] for _ in range(self.n)]
            for cam_src, context_idx_src in pairs:
                context_src, T_src_to_world = self.get_context_and_pose(batch[cam_src], context_idx_src)
                camera_src = self.get_camera(batch, cam_src, context_idx_src, T_src_to_world)

                contexts_src = match_scales(context_src, depths_tgt, self.n, align_corners=self.align_corners)

                mask_src = batch[cam_src]['mask']
                masks_src = match_scales(mask_src, depths_tgt, self.n, align_corners=self.align_corners)
                masks_src_warped = self.warp(camera_tgt, depths_tgt, camera_src, masks_src)
                masks = [(masks_tgt[i] * masks_src_warped[i]).eq(1.0) for i in range(self.n)]

                # Compute multi-scale photometric loss
                contexts_tgt_warped = self.warp(camera_tgt, depths_tgt, camera_src, contexts_src)
                photometric_loss = self.calc_photometric_loss(contexts_tgt_warped, contexts_tgt)

                if return_logs:
                    self.log_images['warped_{}'.format(cam_tgt)].append(
                        (contexts_tgt_warped[0][0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8))
                    self.log_images['warped_{}'.format(cam_tgt)].append(
                        (masks_src_warped[0][0].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8))

                # Apply mask
                for i in range(self.n):
                    photometric_loss[i][~masks[i]] = 0.0
                    photometric_losses[i].append(photometric_loss[i])

                    if return_logs and i == 0:
                        self.log_images['warped_{}'.format(cam_tgt)].append(
                            (torch.sigmoid(photometric_loss[i].detach().float()).repeat(1, 3, 1, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8))
                            # (viz_photo(photometric_loss[0][0, 0].detach().cpu().numpy()) * 255.0).astype(np.uint8))

            if self.automask_loss:
                raw_rgb_key = 'raw_rgb' if self.training else 'rgb'
                context_tgt_all = [raw_rgb for t, raw_rgb in batch[cam_tgt][raw_rgb_key].items() if t != context_idx_tgt]
                for context_tgt_automask in context_tgt_all:
                    contexts = match_scales(context_tgt_automask, depths_tgt, self.n, align_corners=self.align_corners)
                    unwarped_image_loss = self.calc_photometric_loss(contexts, contexts_tgt)
                    for i in range(self.n):
                        photometric_losses[i].append(unwarped_image_loss[i])

            loss, per_pixel_loss = self.reduce_photometric_loss(photometric_losses, debug=return_logs)
            # if loss.isnan().any() or loss.isinf().any() or \
            #     (return_logs and any([l.isnan().any() or l.isinf().any() for l in per_pixel_loss])):
            #     import pdb; pdb.set_trace()

            if return_logs:
                self.log_images['warped_{}'.format(cam_tgt)].append(
                    (torch.sigmoid(per_pixel_loss[0].detach().float()).repeat(1, 3, 1, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8))
                    # (viz_photo(per_pixel_loss[0][0, 0].detach().cpu().numpy()) * 255.0).astype(np.uint8))

            # TODO(soonminh): apply per-pixel weights depending on where the loss comes from
            # if cam_tgt == cam_src:
            #     loss *= self.mono_weight
            # else:
            #     loss *= self.stereo_weight

            # Include smoothness loss if requested
            if self.smooth_loss_weight > 0.0:
                invdepths_tgt = depth2inv(depths_tgt)
                loss += self.calc_smoothness_loss_masked(invdepths_tgt, contexts_tgt, masks_tgt)

            losses[(cam_tgt, context_idx_tgt)] = loss

        loss_sum = sum(losses.values()) / len(losses.values())
        if return_logs:
            log_images = {key: np.vstack(make_same_resolution(images, images[0].shape[:2], self.n))
                            for key, images in self.log_images.items()}
            self.log_images.clear()
        else:
            log_images = {}

        out = {
            'loss': loss_sum.unsqueeze(0),
            'metrics': {},      # TODO(soonminh): why do we need this?
            'log_images': log_images,
        }
        return out
