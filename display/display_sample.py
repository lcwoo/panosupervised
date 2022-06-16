# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.
#   Modified by Soonmin Hwang
#
#   Changes:
#       - Change screens (panes)
#       - Add a "play" mode to sweep all frames and create a video
#           : Move camera based on LiDAR poses
#       - Show predictions
#           : depth map (Push LEFT/RIGHT ARROW to see GTs)
#           : point cloud (Push ENTER to see LiDAR point cloud)
#               + Filtered out top 30% region
#               + Filtered out by self-occlusion mask
#       - Overlay self-occlusion mask to RGB image
#       - High-resolution window to make video quality better
#
#   TODO List
#       - [!!] Re-run inference without "validate_flipped" option for fair comparison
#       - Show camera field-of-views
#       - (Ji's suggestion) Remove 3d points from the camera-overlapped regions (by center lines?)
#       - Camera side-walk in play mode for better visualization
#       - Better filtering of noisy 3D points
#       - Implement slow motion (or stop and side-walk) in play mode
#
#       - (Need idea) better 3D visualization? better view point?
#       - (If needed) show camera frames in 3D screen (maybe once a second?)
#       - Make some settings flexible, such as video fps / initial camera viewpoint
#       - Clean up this script


import cv2
import json
import numpy as np
import os
from PIL import Image

from camviz import BBox3D
from camviz import Camera as CameraCV
from camviz import Draw

from vidar.geometry.camera import Camera
from vidar.geometry.pose import Pose
from vidar.utils.data import make_batch, fold_batch, modrem
from vidar.utils.flip import flip_batch
from vidar.utils.viz import viz_depth, viz_optical_flow, viz_semantic

np.set_printoptions(precision=4)

def change_key(dic, c, n):
    steps = sorted(dic.keys())
    return steps[(steps.index(c) + n) % len(steps)]

def play(draw, data, points, timestamps, t, key, num_cams, cam_orders, cam_colors, all_poses, video_filename=None):

    video_filename += '' if video_filename.endswith('mp4') else '.mp4'

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = draw.wh
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height), True)

    mask = data['mask']
    mask_3ch = 1 - mask.repeat(1, 3, 1, 1)
    mask_3ch[:, 1] *= 0.05

    while t < len(timestamps):

        rgbs = data['rgb'][t]
        rgbs = (rgbs + mask_3ch).clamp(0.0, 1.0)

        for i in range(num_cams):
            ic = cam_orders[i]

            # rgb = data['rgb'][t][ic]
            rgb = rgbs[ic]
            draw.updTexture('cam%d' % i, rgb)

            img_second = data[key][t][ic]
            if key == 'depth':
                img_second = viz_depth(img_second, filter_zeros=True)
            draw.updTexture('depth_pred%d' % i, img_second)

            draw.updBufferf('lidar%d' % i, points[t][i])
            draw.updBufferf('color%d' % i, data['rgb'][t][i])

        draw.screens['wld'].viewer.setPose(all_poses[t])

        draw.clear()

        for i in range(num_cams):
            draw['cam%d%d' % modrem(i, 6)].image('cam%d' % i)
            draw['depth_pred%d%d' % modrem(i, 6)].image('depth_pred%d' % i)
            draw['wld'].size(1).color(cam_colors[i]).points('lidar%d' % i, ('color%d' % i))

        draw.update(10)

        frame = draw.to_image()
        video_writer.write(frame)
        draw.halt(100)

        t = t + 1

    video_writer.release()

    while t not in data['rgb'].keys():
        t = change_key(data['rgb'], t, 0)


def display_sample(data, video_filename, flip=False):

    tasks = ['pred_depth_viz', 'depth', 'rgb', 'fwd_optical_flow', 'bwd_optical_flow','semantic']
    cam_colors = ['red', 'blu', 'gre', 'yel', 'mag', 'cya'] * 100
    cam_orders = (3, 1, 0, 2, 4, 5)
    data = make_batch(data)
    if flip:
        data = flip_batch(data)
    data = fold_batch(data)

    rgb = data['rgb']
    mask = data['mask']
    intrinsics = data['intrinsics']
    depth = data['depth']
    pred_depth_npz = data['pred_depth_npz']
    pose = data['pose']
    pose = Pose.from_dict(pose, to_global=True)
    cam = Camera.from_dict(intrinsics, rgb, pose)

    num_cams = rgb[0].shape[0]
    wh = rgb[0].shape[-2:][::-1]

    keys = [key for key in tasks if key in data.keys()]

    points = {}
    for key, val in cam.items():
        points[key] = cam[key].reconstruct_depth_map(
            depth[key], to_world=True).reshape(num_cams, 3, -1).permute(0, 2, 1)

    pred_points = {}
    for key, val in cam.items():
        reconstructed = cam[key].reconstruct_depth_map(pred_depth_npz[key], to_world=True)
        reconstructed[:, :, :int(0.3 * wh[1])] = 0.0
        reconstructed *= mask
        pred_points[key] = reconstructed.reshape(num_cams, 3, -1).permute(0, 2, 1)

    # draw = Draw((wh[0] * 4, wh[1] * 3), width=1800)
    draw = Draw((wh[0] * 4, wh[1] * 3), width=3600)
    draw.add2DimageGrid('cam', (0.0, 0.0, 1.0, 0.2), n=(1, 6), res=wh)
    draw.add2DimageGrid('depth_pred', (0.0, 0.2, 1.0, 0.4), n=(1, 6), res=wh)
    # draw.add3Dworld('wld', (0.0, 0.4, 1.0, 1.0), pose=cam[0].Tcw.T[0])
    draw.add3Dworld('wld', (0.0, 0.4, 1.0, 1.0))

    draw.addTexture('cam', n=num_cams)
    draw.addTexture('depth_pred', n=num_cams)
    draw.addBuffer3f('lidar', 1000000, n=num_cams)
    draw.addBuffer3f('color', 1000000, n=num_cams)

    with_bbox3d = 'bbox3d' in data
    if with_bbox3d:
        bbox3d_corners = [[BBox3D(b) for b in bb] for bb in data['bbox3d']['corners']]

    with_pointcache = 'pointcache' in data
    if with_pointcache:
        pointcache = np.concatenate([np.concatenate(pp, 0) for pp in data['pointcache']['points']], 0)
        draw.addBufferf('pointcache', pointcache[:, :3])

    camcv = []
    for i in range(num_cams):
        camcv.append({key: CameraCV.from_vidar(val, i) for key, val in cam.items()})

    t, k = 0, 0
    key = keys[k]
    change = True
    color = True
    use_pred_points = True

    timestamps = [f.split('/')[0] + '_' + f.split('/')[-1] for f in data['filename_context'][0]]

    ### Set initial viewpoint
    # 1) Preset 1,
    # pose1 = np.array([
    #     [ 9.2611e-01,  2.5540e-03, -3.7725e-01,  1.2378e+02],
    #     [-1.1802e-02,  9.9968e-01, -2.2206e-02,  1.8982e+01],
    #     [ 3.7707e-01,  2.5017e-02,  9.2585e-01, -5.5307e+02],
    #     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], dtype=np.float32)
    # pose1 = np.array([
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 3],
    #     [1, 0, 0, 0],
    #     [0, 0, 0, 1]], dtype=np.float32)

    # pose1 = np.array([
    #     [0, 1, 0, 0],
    #     [1, 0, 0, 0],
    #     [0, 0, 1, -20],
    #     [0, 0, 0, 1]], dtype=np.float32)

    #     # [ 1.0000e+00, -1.9146e-17,  6.2490e-18,  0.0000e+00],
    #     # [-5.4246e-17,  1.0000e+00, -2.4887e-17,  9.0949e-13],
    #     # [-1.0097e-18,  5.6266e-17,  1.0000e+00, -1.5632e-13],
    #     # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], dtype=np.float32)
    # lidar_pose = data['lidar_pose']
    # # draw.screens['wld'].viewer.setPose(pose1 @ lidar_pose)
    # draw.screens['wld'].viewer.setPose(lidar_pose)

    draw.screens['wld'].viewer.rotateX(-90)
    draw.screens['wld'].viewer.rotateY(90)
    draw.screens['wld'].viewer.rotateZ(180)
    draw.screens['wld'].viewer.rotateY(180)

    # draw.screens['wld'].viewer.rotateX(90)
    # draw.screens['wld'].viewer.rotateY(-90)

    draw.screens['wld'].viewer.rotateX(25)
    # draw.screens['wld'].viewer.translateY(-5)
    draw.screens['wld'].viewer.translateZ(-20)

    pose1 = draw.screens['wld'].viewer.T

    # TODO(sohwang): fall back to the relative pose: lidar_pose_context
    all_poses = np.stack([data['lidar_pose']] + data['lidar_pose_context'], axis=0)
    all_poses = np.matmul(all_poses, pose1)

    draw.screens['wld'].viewer.setPose(all_poses[0])
    draw.screens['wld'].saveViewer()

    tx = 1.0
    rx = 1.0
    ry = 5.0
    points_viz = pred_points if use_pred_points else points
    while draw.input():
        if draw.KEY_W:
            draw.screens['wld'].viewer.rotateY(ry)
            draw.screens['wld'].viewer.rotateX(rx)
            draw.screens['wld'].viewer.translateX(tx)
            # draw.screens['wld'].viewer.translateY(tx)

        if draw.KEY_Q:
            draw.screens['wld'].viewer.translateX(-tx)
            # draw.screens['wld'].viewer.translateY(-tx)
            draw.screens['wld'].viewer.rotateX(-rx)
            draw.screens['wld'].viewer.rotateY(-ry)

        if draw.KEY_1:
            # draw.screens['wld'].viewer.setPose(pose1)
            draw.screens['wld'].reset()
            t = 0
            draw.update(10)
            continue

        if draw.KEY_A:
            # Automatic screenshot
            play(draw, data, points_viz, timestamps, t, key, num_cams, cam_orders, cam_colors, all_poses,
                video_filename)

        if draw.KEY_D:
            import pdb; pdb.set_trace()
            continue
        if draw.KEY_S:
            draw.save(timestamps[t] + '.png')
            continue
        if draw.SPACE:
            color = not color
            change = True
        if draw.RETURN:
            use_pred_points = not use_pred_points
            points_viz = pred_points if use_pred_points else points
            change = True
        if draw.RIGHT:
            change = True
            k = (k + 1) % len(keys)
            while t not in data[keys[k]].keys():
                k = (k + 1) % len(keys)
            key = keys[k]
        if draw.LEFT:
            change = True
            k = (k - 1) % len(keys)
            while t not in data[keys[k]].keys():
                k = (k - 1) % len(keys)
            key = keys[k]
        if draw.UP:
            change = True
            t = change_key(data[key], t, 1)
            while t not in data[keys[k]].keys():
                t = change_key(data[key], t, 1)
            print(t)
        if draw.DOWN:
            change = True
            t = change_key(data[key], t, -1)
            while t not in data[keys[k]].keys():
                t = change_key(data[key], t, -1)
        if change:
            change = False
            rgbs = data['rgb'][t]
            mask_3ch = 1 - mask.repeat(1, 3, 1, 1)
            mask_3ch[:, 1] *= 0.05
            rgbs = (rgbs + mask_3ch).clamp(0.0, 1.0)

            # Camera movement
            draw.screens['wld'].viewer.setPose(all_poses[t])

            for i in range(num_cams):
                ic = cam_orders[i]

                # First row
                # rgb = data['rgb'][t][ic]
                rgb = rgbs[ic]
                draw.updTexture('cam%d' % i, rgb)

                # Second row
                img = data[key][t][ic]
                if key == 'depth':
                    img = viz_depth(img, filter_zeros=True)
                elif key in ['fwd_optical_flow', 'bwd_optical_flow']:
                    img = viz_optical_flow(img)
                draw.updTexture('depth_pred%d' % i, img)
                draw.updBufferf('lidar%d' % i, points_viz[t][i])
                draw.updBufferf('color%d' % i, data['rgb'][t][i])

        draw.clear()
        for i in range(num_cams):
            # draw['cam%d%d' % modrem(i, 2)].image('cam%d' % i)
            draw['cam%d%d' % modrem(i, 6)].image('cam%d' % i)
            draw['depth_pred%d%d' % modrem(i, 6)].image('depth_pred%d' % i)
            # draw['cam%d' % i].image('cam%d' % i)
            draw['wld'].size(1).color(cam_colors[i]).points('lidar%d' % i, ('color%d' % i) if color else None)
            for cam_key, cam_val in camcv[i].items():
                clr = cam_colors[i] if cam_key == t else 'gra'
                tex = 'cam%d' % i if cam_key == t else None
                # draw['wld'].object(cam_val, color=clr, tex=tex)
            if with_bbox3d:
                [[draw['wld'].object(b) for b in bb] for bb in bbox3d_corners]
            if with_pointcache:
                draw['wld'].color('whi').points('pointcache')

        draw.update(30)
