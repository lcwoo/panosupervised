import glob
import numpy as np
import os

from pyquaternion import Quaternion

# from torch.utils.data import Dataset
from vidar.datasets.BaseDataset import BaseDataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
# from packnet_sfm.utils.image import load_image
from vidar.utils.read import read_image

'''
from packnet_sfm.datasets.kitti_dataset_utils import \
    pose_from_oxts_packet, read_calib_file, transform_from_rot_trans
from packnet_sfm.utils.image import load_image
from packnet_sfm.geometry.pose_utils import invert_pose_numpy
'''

########################################################################################################################

SENSORS = {
    'front': 'CAM_FRONT',
    'front_right': 'CAM_FRONT_RIGHT',
    'back_right': 'CAM_BACK_RIGHT',
    'back': 'CAM_BACK',
    'back_left': 'CAM_BACK_LEFT',
    'front_left': 'CAM_FRONT_LEFT',
    'lidar': 'LIDAR_TOP',
}

'''
# Cameras from the stero pair (left is the origin)
IMAGE_FOLDER = {
    'left': 'image_02',
    'right': 'image_03',
}
# Name of different calibration files
CALIB_FILE = {
    'cam2cam': 'calib_cam_to_cam.txt',
    'velo2cam': 'calib_velo_to_cam.txt',
    'imu2velo': 'calib_imu_to_velo.txt',
}
PNG_DEPTH_DATASETS = ['groundtruth']
OXTS_POSE_DATA = 'oxts'
'''

########################################################################################################################
#### FUNCTIONS
########################################################################################################################
'''
def read_npz_depth(file, depth_type):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)[depth_type + '_depth'].astype(np.float32)
    return np.expand_dims(depth, axis=2)

def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(load_image(file), dtype=int)
    assert (np.max(depth_png) > 255), 'Wrong .png depth file'
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return np.expand_dims(depth, axis=2)
'''
########################################################################################################################
#### DATASET
########################################################################################################################

class NuScenesDataset(BaseDataset):
    """
    NuScenes dataset class, using nuscenes-devkit.

    Parameters
    ----------

    """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)

        self.cameras = [SENSORS[c] for c in self.cameras]
        self.sensors = self.cameras + SENSORS['lidar']

        self.nusc = NuScenes(version=split, dataroot=self.path, verbose=True)

        # # Get file list from data
        # from collections import OrderedDict
        self._sample_tokens = list()


        for scene in nusc.scene:
            sample_tok = scene['first_sample_token']
            _tokens = []
            # self._sample_tokens[scene['token']] = []
            while('' != sample_tok):
                # self._sample_tokens[scene['token']].append()
                _tokens.append(sample_tok)
                sample_tok = sample['next']

            self._sample_tokens.append((scene['token'], _tokens))


                # sample = nusc.get('sample', sample_tok)

                # _data_paths = {}
                # for sname in self.sensors:
                #     sample_data_path = nusc.get('sample_data', sample['data'][sname])['filename']
                #     _data_paths[sname].append(os.path.join(self.path, sample_data_path))

                # self._data_paths.append({
                #     'sample_token': sample_tok,
                #     **_data_paths,
                # })
                # sample_tok = sample['next']

        self._num_samples = np.cumsum([0] + [len(sample_tokens) for scene_token, sample_tokens in self._sample_tokens])

    def __len__(self):
        return self._num_samples[-1]

    def _get_intrinsics(self, token):
        calibrated_sensor = self.nusc.get('calibrated_sensor', token)
        return np.array(calibrated_sensor['camera_intrinsic'])

    def _get_extrinsics(self, token):
        calibrated_sensor = self.nusc.get('calibrated_sensor', token)

        extriniscs = np.eye(4)
        extriniscs[:3, :3] = Quaternion(wxyz=calibrated_sensor['rotation'])
        extriniscs[:3, 3:] = calibrated_sensor['translation']
        return extriniscs

    def _get_pose(self, token):
        ego_pose_data = self.nusc.get('ego_pose', token)

        ego_pose = np.eye(4)
        ego_pose[:3, :3] = Quaternion(wxyz=ego_pose_data['rotation'])
        ego_pose[:3, 3:] = ego_pose_data['translation']
        return ego_pose


    def __getitem((self, index):
        sid = np.digitize(index, self._num_samples, right=False) - 1
        fid = index - self._num_samples[sid - 1]

        # Get nusc sample
        scene_token, tokens = self._sample_tokens[sid]
        sample_token = tokens[fid]
        sample = self.nusc.get('sample', sample_token)

        # Loop over all cameras
        samples = []
        for camera in self.cameras:

            sample_data = self.nusc.get('sample_data', sample['data'][camera])

            # Filename
            filename = sample_data['filename']

            # Get sensor info.
            calibrated_sensor =

            # Base data
            data = {
                'idx': index,
                'filename': filename,
                'sensor_name': sample_data['channel'],
                'rgb': read_image(os.path.join(self.path, filename)),
                'intrinsics': self._get_intrinsics(sample_data['calibrated_sensor_token']),
                'extrinsics': self._get_extrinsics(sample_data['calibrated_sensor_token']),
                'ego_pose': self._get_pose(sample_data['ego_pose_token']),
            }

            (WIP)



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

            # If masks are returned
            if self.masks_path is not None:
                sample.update({
                    'mask': read_image(os.path.join(
                        self.masks_path, 'camera_%02d.png' % self.cameras[i]))
                })

            # If depth is returned
            if self.with_depth:
                # Get depth maps
                depth = self.create_proj_maps(
                    filename, i, self.depth_idx, self.depth_type)
                # Include depth map
                sample.update({
                    'depth': {0: depth}
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
                            context=k)
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



########################################################################################################################

    '''
    @staticmethod
    def _get_next_file(idx, file):
        """Get next file given next idx and current file."""
        base, ext = os.path.splitext(os.path.basename(file))
        return os.path.join(os.path.dirname(file), str(idx).zfill(len(base)) + ext)

    @staticmethod
    def _get_parent_folder(image_file):
        """Get the parent folder from image_file."""
        return os.path.abspath(os.path.join(image_file, "../../../.."))

    @staticmethod
    def _get_intrinsics(image_file, calib_data):
        """Get intrinsics from the calib_data dictionary."""
        for cam in ['left', 'right']:
            # Check for both cameras, if found replace and return intrinsics
            if IMAGE_FOLDER[cam] in image_file:
                return np.reshape(calib_data[IMAGE_FOLDER[cam].replace('image', 'P_rect')], (3, 4))[:, :3]

    @staticmethod
    def _read_raw_calib_file(folder):
        """Read raw calibration files from folder."""
        return read_calib_file(os.path.join(folder, CALIB_FILE['cam2cam']))
    '''
########################################################################################################################
#### DEPTH
########################################################################################################################
    def _load_depth(self, depth_file):
        '''return Nx5 pointcloud of float32
        param: depth_file, the nuscenes pcd.bin file (headerless to process)
        '''
        retval = None
        with open(depth_file, 'rb') as f:
            buf = f.read()
            retval = np.frombuffer(buf, dtype=np.float32)

        return retval.reshape((-1,5))


    '''
    def _read_depth(self, depth_file):
        """Get the depth map from a file."""
        if depth_file.endswith('.npz'):
            return read_npz_depth(depth_file, 'velodyne')
        elif depth_file.endswith('.png'):
            return read_png_depth(depth_file)
        else:
            raise NotImplementedError(
                'Depth type {} not implemented'.format(self.depth_type))

    @staticmethod
    def _get_depth_file(image_file, depth_type):
        """Get the corresponding depth file from an image file."""
        for cam in ['left', 'right']:
            if IMAGE_FOLDER[cam] in image_file:
                depth_file = image_file.replace(
                    IMAGE_FOLDER[cam] + '/data', 'proj_depth/{}/{}'.format(
                        depth_type, IMAGE_FOLDER[cam]))
                if depth_type not in PNG_DEPTH_DATASETS:
                    depth_file = depth_file.replace('png', 'npz')
                return depth_file

    def _get_sample_context(self, sample_name,
                            backward_context, forward_context, stride=1):
        """
        Get a sample context

        Parameters
        ----------
        sample_name : str
            Path + Name of the sample
        backward_context : int
            Size of backward context
        forward_context : int
            Size of forward context
        stride : int
            Stride value to consider when building the context

        Returns
        -------
        backward_context : list of int
            List containing the indexes for the backward context
        forward_context : list of int
            List containing the indexes for the forward context
        """
        base, ext = os.path.splitext(os.path.basename(sample_name))
        parent_folder = os.path.dirname(sample_name)
        f_idx = int(base)

        # Check number of files in folder
        if parent_folder in self._cache:
            max_num_files = self._cache[parent_folder]
        else:
            max_num_files = len(glob.glob(os.path.join(parent_folder, '*' + ext)))
            self._cache[parent_folder] = max_num_files

        # Check bounds
        if (f_idx - backward_context * stride) < 0 or (
                f_idx + forward_context * stride) >= max_num_files:
            return None, None

        # Backward context
        c_idx = f_idx
        backward_context_idxs = []
        while len(backward_context_idxs) < backward_context and c_idx > 0:
            c_idx -= stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                backward_context_idxs.append(c_idx)
        if c_idx < 0:
            return None, None

        # Forward context
        c_idx = f_idx
        forward_context_idxs = []
        while len(forward_context_idxs) < forward_context and c_idx < max_num_files:
            c_idx += stride
            filename = self._get_next_file(c_idx, sample_name)
            if os.path.exists(filename):
                forward_context_idxs.append(c_idx)
        if c_idx >= max_num_files:
            return None, None

        return backward_context_idxs, forward_context_idxs

    def _get_context_files(self, sample_name, idxs):
        """
        Returns image and depth context files

        Parameters
        ----------
        sample_name : str
            Name of current sample
        idxs : list of idxs
            Context indexes

        Returns
        -------
        image_context_paths : list of str
            List of image names for the context
        depth_context_paths : list of str
            List of depth names for the context
        """
        image_context_paths = [self._get_next_file(i, sample_name) for i in idxs]
        return image_context_paths, None
    '''
########################################################################################################################
#### POSE
########################################################################################################################
    '''
    def _get_imu2cam_transform(self, image_file):
        """Gets the transformation between IMU an camera from an image file"""
        parent_folder = self._get_parent_folder(image_file)
        if image_file in self.imu2velo_calib_cache:
            return self.imu2velo_calib_cache[image_file]

        cam2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['cam2cam']))
        imu2velo = read_calib_file(os.path.join(parent_folder, CALIB_FILE['imu2velo']))
        velo2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['velo2cam']))

        velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
        cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))

        imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat
        self.imu2velo_calib_cache[image_file] = imu2cam
        return imu2cam

    @staticmethod
    def _get_oxts_file(image_file):
        """Gets the oxts file from an image file."""
        # find oxts pose file
        for cam in ['left', 'right']:
            # Check for both cameras, if found replace and return file name
            if IMAGE_FOLDER[cam] in image_file:
                return image_file.replace(IMAGE_FOLDER[cam], OXTS_POSE_DATA).replace('.png', '.txt')
        # Something went wrong (invalid image file)
        raise ValueError('Invalid KITTI path for pose supervision.')

    def _get_oxts_data(self, image_file):
        """Gets the oxts data from an image file."""
        oxts_file = self._get_oxts_file(image_file)
        if oxts_file in self.oxts_cache:
            oxts_data = self.oxts_cache[oxts_file]
        else:
            oxts_data = np.loadtxt(oxts_file, delimiter=' ', skiprows=0)
            self.oxts_cache[oxts_file] = oxts_data
        return oxts_data

    def _get_pose(self, image_file):
        """Gets the pose information from an image file."""
        if image_file in self.pose_cache:
            return self.pose_cache[image_file]
        # Find origin frame in this sequence to determine scale & origin translation
        base, ext = os.path.splitext(os.path.basename(image_file))
        origin_frame = os.path.join(os.path.dirname(image_file), str(0).zfill(len(base)) + ext)
        # Get origin data
        origin_oxts_data = self._get_oxts_data(origin_frame)
        lat = origin_oxts_data[0]
        scale = np.cos(lat * np.pi / 180.)
        # Get origin pose
        origin_R, origin_t = pose_from_oxts_packet(origin_oxts_data, scale)
        origin_pose = transform_from_rot_trans(origin_R, origin_t)
        # Compute current pose
        oxts_data = self._get_oxts_data(image_file)
        R, t = pose_from_oxts_packet(oxts_data, scale)
        pose = transform_from_rot_trans(R, t)
        # Compute odometry pose
        imu2cam = self._get_imu2cam_transform(image_file)
        odo_pose = (imu2cam @ np.linalg.inv(origin_pose) @
                    pose @ np.linalg.inv(imu2cam)).astype(np.float32)
        # Cache and return pose
        self.pose_cache[image_file] = odo_pose
        return odo_pose
    '''
########################################################################################################################


    def __getitem__(self, idx):
        """Get dataset sample given an index."""

        sample = {'idx': idx}
        # add camera images
        sample.update({ cname: load_image(self._data_paths[cname][idx]) for cname in self._cameras})
        # add LIDAR
        sample.update({self.LIDAR_TOP: LidarPointCloud.from_file(self._data_paths[self.LIDAR_TOP][idx])})

        '''
        sample = {
            'idx': idx,
            # 'sample_token': self.sample_paths[idx],
            #'rgb': load_image(self.sample_tokens[idx]),
        }
        # Add intrinsics
        parent_folder = self._get_parent_folder(self.sample_tokens[idx])
        if parent_folder in self.calibration_cache:
            c_data = self.calibration_cache[parent_folder]
        else:
            c_data = self._read_raw_calib_file(parent_folder)
            self.calibration_cache[parent_folder] = c_data
        sample.update({
            'intrinsics': self._get_intrinsics(self.sample_tokens[idx], c_data),
        })

        # Add pose information if requested
        if self.with_pose:
            sample.update({
                'pose': self._get_pose(self.sample_tokens[idx]),
            })

        # Add depth information if requested
        if self.with_depth:
            sample.update({
                'depth': self._read_depth(self._get_depth_file(
                    self.sample_tokens[idx], self.depth_type)),
            })

        # Add input depth information if requested
        if self.with_input_depth:
            sample.update({
                'input_depth': self._read_depth(self._get_depth_file(
                    self.sample_tokens[idx], self.input_depth_type)),
            })

        # Add context information if requested
        if self.with_context:
            # Add context images
            all_context_idxs = self.backward_context_paths[idx] + \
                               self.forward_context_paths[idx]
            image_context_paths, _ = \
                self._get_context_files(self.sample_tokens[idx], all_context_idxs)
            image_context = [load_image(f) for f in image_context_paths]
            sample.update({
                'rgb_context': image_context
            })
            # Add context poses
            if self.with_pose:
                first_pose = sample['pose']
                image_context_pose = [self._get_pose(f) for f in image_context_paths]
                image_context_pose = [invert_pose_numpy(context_pose) @ first_pose
                                      for context_pose in image_context_pose]
                sample.update({
                    'pose_context': image_context_pose
                })

        # Apply transformations
        if self.data_transform:
            sample = self.data_transform(sample)
        '''

        # Return sample
        return sample

########################################################################################################################