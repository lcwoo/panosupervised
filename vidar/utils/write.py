# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import os
import pickle as pkl

import cv2
import numpy as np
import torchvision.transforms as transforms

from vidar.utils.decorators import multi_write
from vidar.utils.types import is_tensor, is_numpy
from vidar.utils.viz import viz_depth, viz_inv_depth
# from vidar.utils.write import write_image

def create_folder(filename):
    """Create a new folder if it doesn't exist"""
    if '/' in filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)


def write_pickle(filename, data):
    """
    Write a pickle file

    Parameters
    ----------
    filename : String
        File where the pickle file will be saved
    data : Value
        Data to be saved
    """
    create_folder(filename)
    if not filename.endswith('.pkl'):
        filename = filename + '.pkl'
    pkl.dump(data, open(filename, 'wb'))


def write_npz(filename, data):
    """
    Write a numpy compressed file

    Parameters
    ----------
    filename : String
        File where the numpy file will be saved
    data : Value
        Data to be saved
    """
    np.savez_compressed(filename, **data)


@multi_write
@multi_write
def write_image(filename, image):
    """
    Write an image to file

    Parameters
    ----------
    filename : String
        File where image will be saved
    image : np.Array [H,W,3]
        RGB image
    """
    # Create folder if it doesn't exist
    create_folder(filename)
    # If image is a tensor
    if is_tensor(image):
        if len(image.shape) == 4:
            image = image[0]
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
        cv2.imwrite(filename, image[:, :, ::-1] * 255)
    # If image is a numpy array
    elif is_numpy(image):
        cv2.imwrite(filename, image[:, :, ::-1] * 255)
    # Otherwise, assume it's a PIL image
    else:
        image.save(filename)


@multi_write
def write_depth(filename, depth, intrinsics=None):
    """
    Write a depth map to file, and optionally its corresponding intrinsics.

    Parameters
    ----------
    filename : String
        File where depth map will be saved (.npz or .png)
    depth : np.Array or torch.Tensor
        Depth map [H,W]
    intrinsics : np.Array
        Optional camera intrinsics matrix [3,3]
    """
    # If depth is a tensor
    if is_tensor(depth):
        depth = depth.detach().squeeze().cpu().numpy()
    # If intrinsics is a tensor
    if is_tensor(intrinsics):
        intrinsics = intrinsics.detach().cpu().numpy()
    # If we are saving as a .npz
    if filename.endswith('.npz'):
        np.savez_compressed(filename, depth=depth, intrinsics=intrinsics)
    # If we are saving as a .png
    elif filename.endswith('.png'):
        depth = transforms.ToPILImage()((depth * 256).astype(np.uint8))
        depth.save(filename)
    # Something is wrong
    else:
        raise NotImplementedError('Depth filename not valid.')


@multi_write
def write_optical_flow(filename, optflow):
    """
    Write a depth map to file, and optionally its corresponding intrinsics.

    Parameters
    ----------
    filename : String
        File where depth map will be saved (.npz or .png)
    optflow : np.Array or torch.Tensor
        Optical flow map [H,W]
    """
    # If depth is a tensor
    if is_tensor(optflow):
        optflow = optflow.detach().squeeze().cpu().numpy()
    # If we are saving as a .npz
    if filename.endswith('.npz'):
        np.savez_compressed(filename, optflow=optflow)
    # Something is wrong
    else:
        raise NotImplementedError('Optical flow filename not valid.')


def draw_inputs(batch):
    """Useful for debugging to see the input data right before feeding to the network"""

    camera_order = ['camera_07', 'camera_05', 'camera_01', 'camera_06', 'camera_08', 'camera_09']
    float_tensor_to_uint8_numpy = lambda x: (x[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    scene, *_, timestamp = batch['camera_pano']['filename'][0].split('/')
    filename = f'{scene}_{timestamp}'

    gt_pano_depth = (viz_depth(batch['camera_pano']['depth'][0][0]) * 255.0).astype(np.uint8)
    h_viz, w_viz = gt_pano_depth.shape[:2]

    raw_rgb = [float_tensor_to_uint8_numpy(batch[c]['raw_rgb'][0]) for c in camera_order]
    raw_rgb = np.hstack(raw_rgb)
    _h, _w = raw_rgb.shape[:2]
    raw_rgb = cv2.resize(raw_rgb, (0, 0), fx=w_viz/_w, fy=w_viz/_w)
    # cv2.imwrite(f'{filename}_raw_rgb.png', raw_rgb[:,:,(2,1,0)])

    aug_rgb = [float_tensor_to_uint8_numpy(batch[c]['rgb'][0]) for c in camera_order]
    aug_rgb = np.hstack(aug_rgb)
    _h, _w = aug_rgb.shape[:2]
    aug_rgb = cv2.resize(aug_rgb, (0, 0), fx=w_viz/_w, fy=w_viz/_w)
    # cv2.imwrite(f'{filename}_aug_rgb.png', aug_rgb[:,:,(2,1,0)])

    # cv2.imwrite(f'{filename}_frame.png', np.vstack([raw_rgb, aug_rgb, gt_pano_depth])[:,:,(2,1,0)])
    return np.vstack([raw_rgb, aug_rgb, gt_pano_depth])[:,:,(2,1,0)]
