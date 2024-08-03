# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.
import copy
import os
import pickle

import numpy as np
import torch

from dgp.utils.camera import Camera
from dgp.utils.pose import Pose

from vidar.datasets.BaseDataset import BaseDataset
from vidar.datasets.utils.misc import stack_sample, make_relative_pose
from vidar.utils.data import make_list
from vidar.utils.read import read_image
from vidar.utils.types import is_str


def load_from_file(filename, *keys):
    """Load data cache from a file"""
    if is_str(keys):
        keys = [keys]
    data = np.load(filename, allow_pickle=True)
    out = [data[k] for k in keys]
    if len(out[0].shape) == 0:
        out = None
    return out[0] if len(keys) == 1 else out


def save_to_file(filename, data_dict):
    """Save data to a cache file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(filename, **data_dict)


def generate_proj_maps(camera, Xw, Xl, shape):
    """Render pointcloud on image.

    Parameters
    ----------
    camera: Camera
        Camera object with appropriately set extrinsics wrt world.
    Xw: np.Array
        3D point cloud (x, y, z) in the world coordinate. [N,3]
    Xl: np.Array
        3D point cloud (x, y, z) in the lidar coordinate. [N,3]
    shape: np.Array
        Output depth image shape [H, W]

    Returns
    -------
    depth: np.Array
        Rendered depth image
    """
    assert len(shape) == 2, 'Shape needs to be 2-tuple.'

    # Move point cloud to the camera's (C) reference frame from the world (W)
    ones = torch.ones(Xw.shape[0], 1, dtype=torch.float64).to(torch.device('cuda'))
    homogeneous_Xw = torch.cat((Xw, ones), dim=1)

    camera_pose = torch.tensor(camera.p_cw.matrix, dtype=torch.float64).to(torch.device('cuda'))
    Xc = torch.matmul(camera_pose, homogeneous_Xw.T)[:3, :].T
    # Project the points as if they were in the camera's frame of reference
    uv = Camera(K=camera.K).project_tensor(Xc).long()
    
    # Colorize the point cloud based on depth
    z_c = Xc[:, 2]

    # Create an empty image to overlay
    H, W = shape
    proj_depth = torch.zeros((H, W), dtype=torch.float64).to(uv.device)
    in_view = (uv >= 0).all(dim=1) & (uv[:, 0] < W) & (uv[:, 1] < H) & (z_c > 0)
    uv, z_c = uv[in_view], z_c[in_view]
    proj_depth[uv[:, 1], uv[:, 0]] = z_c

    # Calculate yaw angle in LiDAR coordinate
    xx = Xl[in_view][:, 0]
    yy = Xl[in_view][:, 1]
    yaw = torch.atan2(xx, -yy + 1e-6)

    # HACK(soonminh): Reverse yaw to make it clockwise and add pi to start from backward
    # yaw = -yaw
    yaw = yaw.to(torch.float64)
    proj_angle = torch.zeros((H, W), dtype=torch.float64, device=uv.device)
    proj_angle[uv[:, 1], uv[:, 0]] = yaw
    return proj_depth, proj_angle


    # ### Calculate polar/azimuth angles in LiDAR coordinate
    # # [LiDAR coordinate convention] https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # #   (X, Y, Z) = (N, W, U), where northing (N), westing(W), and upwardness (U)
    # #   polar angle (theta): measured from a fixed zenith direction
    # #   azimuth angle (phi): measured from negative X-axis (southing, S) on a reference (XY-) plane
    # x = Xl[in_view][:, 0]
    # y = Xl[in_view][:, 1]
    # z = Xl[in_view][:, 2]
    # r = np.linalg.norm(Xl[in_view], 2, axis=1)
    # theta = np.arccos(z / (r + 1e-6))
    # phi = np.arctan2(y, x + 1e-6)

    # # Make phi positive angle
    # phi = phi + np.pi
    # # HACK(soonminh): Reverse yaw to make it clockwise and add pi to start from backward


    # proj_angle = np.zeros((H, W), dtype=np.float32)
    # proj_angle[uv[:, 1], uv[:, 0]] = theta
    # return proj_depth, proj_angle


class OuroborosDataset(BaseDataset):
    """
    DGP dataset class

    Parameters
    ----------
    path : String
        Path to the dataset
    split : String {'train', 'val', 'test'}
        Which dataset split to use
    cameras : list[String]
        Which cameras to get information from
    depth_type : String
        Which lidar will be used to generate ground-truth information
    input_depth_type : String
        Which lidar will be used as input to the networks
    with_pose : Bool
        If enabled pose estimates are also returned
    with_extra_context : Bool
        If enabled extra context information (e.g. depth, semantic, instance) are also returned
    back_context : Int
        Size of the backward context
    forward_context : Int
        Size of the forward context
    data_transform : Function
        Transformations applied to the sample
    dataset : String ['synchronized', 'parallel_domain']
        Which dataset will be used
    only_cache : Bool
        Only use cached pointcloud information, without loading the sensor
    do_stack_samples : bool
        If enabled returned samples are stacked
    """
    def __init__(self, split, tag=None,
                 depth_type=None, input_depth_type=None,
                 masks=None, do_stack_samples=True, **kwargs):
        super().__init__(**kwargs)
        self.tag = 'ouroboros' if tag is None else tag

        cameras = [c if is_str(c) else 'camera_%02d' % c for c in self.cameras]

        # Store variables
        self.split = split
        self.dataset_idx = 0
        self.sensors = list(cameras)

        # Store task information
        self.depth_type = depth_type
        self.input_depth_type = input_depth_type
        self.only_cache = False

        self.masks_path = masks

        # Add requested annotations
        requested_annotations = []

        # Add depth sensor
        if self.with_depth and not self.only_cache and \
                self.depth_type != 'zbuffer':
            self.sensors.append(depth_type)
        self.depth_idx = len(self.sensors) - 1

        # Add input depth sensor
        if self.with_input_depth and not self.only_cache and \
                self.input_depth_type != 'zbuffer' and \
                self.input_depth_type != self.depth_type:
            self.sensors.append(input_depth_type)
        self.input_depth_idx = len(self.sensors) - 1

        # Add radar sensor
        if self.with_radar:
            self.sensors.append('radar')
        self.radar_idx = len(self.sensors) - 1
 

        # Choose which dataset to use
        if not self.virtual:
            from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
            dataset = SynchronizedSceneDataset
            extra_args = {}
        else:
            from dgp.datasets.pd_dataset import ParallelDomainSceneDataset
            dataset = ParallelDomainSceneDataset
            extra_args = {
                'use_virtual_camera_datums': False,
            }

        # Initialize chosen dataset
        self.dataset = dataset(
            scene_dataset_json=self.path,
            split=split,
            datum_names=self.sensors,
            backward_context=self.bwd_context,
            forward_context=self.fwd_context,
            requested_annotations=requested_annotations,
            only_annotated_datums=False,
            **extra_args,
        )

        self.do_stack_samples = do_stack_samples

    def depth_to_points(self, depth, datum_idx, coord='world'):
        """
        Unproject depth from a camera's perspective into a world-frame pointcloud

        Parameters
        ----------
        depth : np.Array
            Depth map to be lifted [H,W]
        datum_idx : Int
            Index of the camera
        coord: String (world, ego, cam)
            Coordinate of the output points

        Returns
        -------
        pointcloud : np.Array
            Lifted 3D pointcloud [Nx3]
        """
        assert coord in ('world', 'ego', 'cam')
        # Access data
        intrinsics = self.get_current('intrinsics', datum_idx)
        pose = self.get_current('pose' if coord == 'world' else 'extrinsics', datum_idx)
        pose = copy.deepcopy(pose)
        # Create pixel grid for 3D unprojection
        h, w = depth.shape[:2]
        uv = np.mgrid[:w, :h].transpose(2, 1, 0).reshape(-1, 2).astype(np.float32)
        # Unproject grid to 3D in the camera frame of reference
        pcl = Camera(K=intrinsics).unproject(uv) * depth.reshape(-1, 1)

        # return pose * pcl if coord != 'cam' else pcl
        if coord == 'cam':
            pose.tvec[...] = 0.0
        return pose * pcl

    def create_camera(self, datum_idx, context=None):
        """
        Create current camera

        Parameters
        ----------
        datum_idx : Int
            Index of the camera
        context : Int
            Context value for choosing current of reference information

        Returns
        -------
        camera : Camera
            DGP camera
        """
        camera_pose = self.get_current_or_context('pose', datum_idx, context)
        camera_intrinsics = self.get_current_or_context('intrinsics', datum_idx, context)
        return Camera(K=camera_intrinsics, p_cw=camera_pose.inverse())

    def create_camera_rays(self, camera_idx, distance=10.0, coord='ego'):
        image_shape = self.get_current('rgb', camera_idx).size[::-1]
        pcl = self.depth_to_points(distance * np.ones(image_shape), camera_idx, coord)

        ### Calculate polar/azimuth angles in LiDAR coordinate
        # [LiDAR coordinate convention] https://en.wikipedia.org/wiki/Spherical_coordinate_system
        #   (X, Y, Z) = (N, W, U), where northing (N), westing(W), and upwardness (U)
        #   polar angle (theta): measured from a fixed zenith direction
        #   azimuth angle (phi): measured from negative X-axis (southing, S) on a reference (XY-) plane
        r = np.linalg.norm(pcl, 2, axis=1)
        x, y, z = pcl.T

        # measured from Z-axis
        theta = np.arccos(z / (r + 1e-6))
        # measured from X-axis, counterclockwise
        phi = np.arctan2(y, x + 1e-6)
        # Make phi positive/clockwise angle
        phi = -phi + np.pi
        import ipdb;ipdb.set_trace()
        rays = np.stack([theta, phi], axis=0).reshape(2, *image_shape).astype(np.float32)
        return rays

    def create_proj_maps(self, filename, camera_idx, depth_idx, depth_type,
                         world_points=None, context=None):
        """
        Creates the depth map for a camera by projecting LiDAR information.
        It also caches the depth map following DGP folder structure, so it's not recalculated

        Parameters
        ----------
        filename : String
            Filename used for loading / saving
        camera_idx : Int
            Camera sensor index
        depth_idx : Int
            Depth sensor index
        depth_type : String
            Which depth type will be loaded
        world_points : np.Array [Nx3]
            Points that will be projected (optional)
        context : Int
            Context value for choosing current of reference information

        Returns
        -------
        depth : np.Array
            Depth map for that datum in that sample [H,W]
        """
        # If we want the z-buffer (simulation)
        if depth_type == 'zbuffer':
            sensor_name = self.get_current('datum_name', camera_idx)
            filename = filename.replace(self.sensors[camera_idx], sensor_name)
            filename = '{}/{}.npz'.format(
                os.path.dirname(self.path), filename.format('depth'))
            return np.load(filename)['data'], None, None
        # Otherwise, we want projected information
        filename_depth = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('depth/{}'.format(depth_type)))
        # Load and return if exists
        try:
            # Get cached depth map
            # depth = load_from_file(filename_depth, 'depth')
            depth, angle = load_from_file(filename_depth, ['depth', 'angle'])
            return depth, angle
        except:
            pass
        # Calculate world points if needed
        if world_points is None:
            # Get lidar information
            lidar_pose = torch.from_numpy(self.get_current_or_context('pose', depth_idx, context).matrix).to(torch.device('cuda'))

            lidar_points = self.get_current_or_context('point_cloud', depth_idx, context)
            lidar_points = torch.from_numpy(lidar_points).float().to(torch.device('cuda'))

            ones = torch.ones((lidar_points.size(0), 1)).to(torch.device('cuda'))
            homogeneous_lidar_points = torch.cat((lidar_points, ones), dim=1).to(dtype=torch.float64)
            world_points = lidar_pose @ homogeneous_lidar_points.T
            world_points = world_points[:3, :].T

        # Create camera
        camera = self.create_camera(camera_idx, context)
        image_shape = self.get_current_or_context('rgb', camera_idx, context).size[::-1]
        # Generate depth maps
        depth, angle  = generate_proj_maps(camera, world_points, lidar_points, image_shape)
        # Save depth map
        depth = depth.detach().cpu().numpy()
        angle = angle.detach().cpu().numpy()
        
        save_to_file(filename_depth, {'depth': depth, 'angle': angle})
        # Return depth
        return depth, angle


    def get_current(self, key, sensor_idx, as_dict=False):
        """Return current timestep of a key from a sensor"""
        current = self.sample_dgp[self.bwd_context][sensor_idx][key]
        return current if not as_dict else {0: current}

    def get_backward(self, key, sensor_idx):
        """Return backward timesteps of a key from a sensor"""
        return [] if self.bwd_context == 0 else \
            [self.sample_dgp[i][sensor_idx][key] for i in range(0, self.bwd_context)]

    def get_forward(self, key, sensor_idx):
        """Return forward timesteps of a key from a sensor"""
        return [] if self.fwd_context == 0 else \
            [self.sample_dgp[i][sensor_idx][key]
             for i in range(self.bwd_context + 1,
                            self.bwd_context + self.fwd_context + 1)]

    def get_context(self, key, sensor_idx, as_dict=False):
        """Get both backward and forward contexts"""
        context = self.get_backward(key, sensor_idx) + self.get_forward(key, sensor_idx)
        if not as_dict:
            return context
        else:
            return {key: val for key, val in zip(self.context, context)}

    def get_current_or_context(self, key, sensor_idx, context=None, as_dict=False):
        """Return current or context information for a given key and sensor index"""
        if context is None:
            return self.get_current(key, sensor_idx, as_dict=as_dict)
        else:
            return self.get_context(key, sensor_idx, as_dict=as_dict)[context]

    def has_dgp_key(self, key, sensor_idx):
        """Returns True if the DGP sample contains a certain key"""
        return key in self.sample_dgp[self.bwd_context][sensor_idx].keys()

    def get_filename(self, sample_idx, datum_idx, context=0):
        """
        Returns the filename for an index, following DGP structure

        Parameters
        ----------
        sample_idx : Int
            Sample index
        datum_idx : Int
            Datum index
        context : Int
            Context offset for the sample

        Returns
        -------
        filename : String
            Filename for the datum in that sample
        """
        scene_idx, sample_idx_in_scene, _ = self.dataset.dataset_item_index[sample_idx]
        scene_dir = self.dataset.scenes[scene_idx].directory
        filename = self.dataset.get_datum(
            scene_idx, sample_idx_in_scene + context, self.sensors[datum_idx]).datum.image.filename
        return os.path.splitext(os.path.join(os.path.basename(scene_dir),
                                             filename.replace('rgb', '{}')))[0]

    def __len__(self):
        """Length of dataset"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get dataset sample"""

        # Get DGP sample (if single sensor, make it a list)
        self.sample_dgp = self.dataset[idx]
        self.sample_dgp = [make_list(sample) for sample in self.sample_dgp]

        # Reorganize sensors to the right order
        sensor_names = [self.get_current('datum_name', i).lower() for i in range(len(self.sensors))]
        indexes = [sensor_names.index(v) for v in self.sensors]
        self.sample_dgp = [[s[idx] for idx in indexes] for s in self.sample_dgp]

        # Loop over all cameras
        samples = []
        for i in range(self.num_cameras):

            # Filename
            filename = self.get_filename(idx, i)

            # Base sample
            sample = {
                'idx': idx,
                'tag': self.tag,
                'filename': self.relative_path({0: filename}),
                'splitname': '%s_%010d' % (self.split, idx),
                'sensor_name': self.get_current('datum_name', i),
            }

            # Image and intrinsics
            sample.update({
                'rgb': self.get_current('rgb', i, as_dict=True),
                'intrinsics': self.get_current('intrinsics', i, as_dict=True),
            })

            # Rays
            if self.with_rays:
                rays = self.create_camera_rays(i)
                embedding = np.stack([
                    np.sin(rays[0]),
                    np.cos(rays[0]),
                    np.sin(rays[1]),
                    np.cos(rays[1]),
                ], axis=0)
                sample.update({
                    'rays': {0: rays},
                    'rays_embedding': {0: embedding},
                })

            # If masks are returned
            if self.masks_path is not None:
                sample.update({
                    'mask': read_image(os.path.join(
                        '/home/lcw/workspace/panosupervised/data/masks/ddad', '%02d.png' % self.cameras[i]))
                })

            # If depth is returned
            if self.with_depth:
                # Get depth maps
                depth, angle = self.create_proj_maps(
                    filename, i, self.depth_idx, self.depth_type)
                # Include depth map
                sample.update({
                    'depth': {0: depth},
                    'angle': {0: angle},
                })

            # If input depth is returned
            if self.with_input_depth:
                sample.update({
                    'input_depth': {0: self.create_proj_maps(
                        filename, i, self.input_depth_idx, self.input_depth_type)[0]}
                })

            # If pose is returned
            if self.with_pose:
                sample.update({
                    'extrinsics': {key: val.inverse().matrix for key, val in
                                   self.get_current('extrinsics', i, as_dict=True).items()},
                    'pose': {key: val.inverse().matrix for key, val in
                             self.get_current('pose', i, as_dict=True).items()},
                })

            # If context is returned
            if self.with_context:

                # Include context images
                sample['rgb'].update(self.get_context('rgb', i, as_dict=True))

                # Create contexts filenames if extra context is required
                filename_context = []
                for context in range(-self.bwd_context, 0):
                    filename_context.append(self.get_filename(idx, i, context))
                for context in range(1, self.fwd_context + 1):
                    filename_context.append(self.get_filename(idx, i, context))
                sample['filename_context'] = filename_context

                # If context pose is returned
                if self.with_pose:
                    # Get original values to calculate relative motion
                    inv_orig_extrinsics = Pose.from_matrix(sample['extrinsics'][0]).inverse()
                    sample['extrinsics'].update(
                        {key: (inv_orig_extrinsics * val.inverse()).matrix for key, val in zip(
                            self.context, self.get_context('extrinsics', i))})
                    sample['pose'].update(
                        {key: (val.inverse()).matrix for key, val in zip(
                            self.context, self.get_context('pose', i))})

                # If context depth is returned
                if self.with_depth_context:
                    depth_context = [
                        self.create_proj_maps(
                            filename, i, self.depth_idx, self.depth_type,
                            context=k)[0]
                        for k, filename in enumerate(filename_context)]
                    sample['depth'].update(
                        {key: val for key, val in zip(
                            self.context, [dsf for dsf in depth_context])})


            samples.append(sample)

        # Make relative poses
        samples = make_relative_pose(samples)

        # Add LiDAR information

        lidar_sample = {}
        if self.with_lidar:

            # Include pointcloud information
            lidar_sample.update({
                'lidar_pointcloud': self.get_current('point_cloud', self.depth_idx),
            })

            # If pose is included
            if self.with_pose:
                lidar_sample.update({
                    'lidar_extrinsics': self.get_current('extrinsics', self.depth_idx).matrix,
                    'lidar_pose': self.get_current('pose', self.depth_idx).matrix,
                })

            # If extra context is included
            if self.with_extra_context:
                lidar_sample['lidar_context'] = self.get_context('point_cloud', self.depth_idx)
                # If context pose is included
                if self.with_pose:
                    # Get original values to calculate relative motion
                    orig_extrinsics = Pose.from_matrix(lidar_sample['lidar_extrinsics'])
                    orig_pose = Pose.from_matrix(lidar_sample['lidar_pose'])
                    lidar_sample.update({
                        'lidar_extrinsics_context':
                            [(orig_extrinsics.inverse() * extrinsics).inverse().matrix
                             for extrinsics in self.get_context('extrinsics', self.depth_idx)],
                        # 'lidar_pose_context':
                        #     [(orig_pose.inverse() * pose).inverse().matrix
                        #      for pose in self.get_context('pose', self.depth_idx)],
                        'lidar_pose_context':
                            [pose.matrix for pose in self.get_context('pose', self.depth_idx)],
                    })


        # Apply same data transformations for all sensors
        if self.data_transform:
            samples = self.data_transform(samples)
            # lidar_sample = self.data_transform(lidar_sample)

        # Return sample (stacked if necessary)
        return stack_sample(samples, lidar_sample) if self.do_stack_samples else (samples, lidar_sample)
