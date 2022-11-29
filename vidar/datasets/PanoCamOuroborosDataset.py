import numpy as np
import os
import torch

from vidar.datasets.augmentations.resize import resize_npy_preserve
from vidar.geometry.camera_pano import PanoCamera
from vidar.datasets.OuroborosDataset import load_from_file, save_to_file, generate_proj_maps
from vidar.datasets.OuroborosDataset import OuroborosDataset
from vidar.datasets.utils.misc import stack_sample


PANO_CAMERA_NAME = 'camera_pano'


def generate_pano_proj_maps(camera, Xw, Xl):
    """Render pointcloud on pano image.

    Parameters
    ----------
    camera: PanoCamera
        Camera object with appropriately set extrinsics wrt world.
    Xw: np.Array
        3D point cloud (x, y, z) in the world coordinate. [N,3]
    Xl: np.Array
        3D point cloud (x, y, z) in the lidar coordinate. [N,3]
    Returns
    -------
    depth: np.Array
        Rendered pano depth image
    """
    # Project the points
    uv_tensor, rho_tensor = camera.project_points(Xw, normalize=False, return_z=True)
    uv = uv_tensor[0].numpy().astype(int)
    # Colorize the point cloud based on depth
    rho = rho_tensor[0].numpy()

    # Create an empty image to overlay
    H, W = camera.hw
    proj_depth = np.zeros((H, W), dtype=np.float32)
    in_view = np.logical_and.reduce([(uv >= 0).all(axis=1), uv[:, 0] < W, uv[:, 1] < H, rho > 0])
    uv, rho = uv[in_view], rho[in_view]

    # TODO(sohwang): this is not enough, we need meshes and filter points by z-buffer.
    # Sort by distance to pick closest one if multiple LiDAR points are projected onto a single pixel
    order = np.argsort(rho)[::-1]
    uv = uv[order]
    rho = rho[order]
    proj_depth[uv[:, 1], uv[:, 0]] = rho

    # Calculate yaw angle in LiDAR coordinate
    xx = Xl[in_view][:, 0]
    yy = Xl[in_view][:, 1]
    yaw = np.arctan2(yy, xx + 1e-6)

    # HACK(soonminh): Reverse yaw to make it clockwise and add pi to start from backward
    yaw = -yaw + np.pi

    proj_angle = np.zeros((H, W), dtype=np.float32)
    proj_angle[uv[:, 1], uv[:, 0]] = yaw

    return proj_depth, proj_angle


########################################################################################################################
#### DATASET
########################################################################################################################
class MultiCamOuroborosDataset(OuroborosDataset):
    """
    MultiCamOuroborosDataset dataset class for MultiCam models, which inherits OuroborosDataset.
    This class returns per-camera dictionary of batches for MultiCam models
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, do_stack_samples=False)

    def __getitem__(self, idx):
        samples, lidar_sample = super().__getitem__(idx)

        samples_dict = {sample['sensor_name'].lower(): stack_sample([sample]) for sample in samples}
        samples_dict['idx'] = samples[0]['idx']
        samples_dict.update(lidar_sample)
        return samples_dict


class PanoCamOuroborosDataset(MultiCamOuroborosDataset):
    """
    PanoCamOuroborosDataset dataset class for PanoCam models, which inherits MultiCamOuroborosDataset.
    This class returns per-camera dictionary of batches for PanoCam models and PanoDepth GT for evaluation.
    """
    def __init__(self, *args, **kwargs):
        pano_cfg = kwargs.pop('pano_cam_config')
        super().__init__(*args, **kwargs)

        self.pano_name = pano_cfg.name
        self.pano_cfg = pano_cfg.dict

    def create_pano_proj_maps(self, filename, K, hw, Twc, depth_idx, depth_type):
        """
        Creates the depth map for a camera by projecting LiDAR information.
        It also caches the depth map following DGP folder structure, so it's not recalculated

        Parameters
        ----------
        filename : String
            Filename used for loading / saving
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
        filename_depth = '{}/{}.npz'.format(
            os.path.dirname(self.path), filename.format('depth/{}'.format(depth_type)))
        # Load and return if exists
        try:
            # Get cached depth map
            # depth, depth_yaw = load_from_file(filename_depth, 'depth', 'depth_yaw')
            depth, angle = load_from_file(filename_depth, ['depth', 'angle'])
            return depth, angle
        except:
            pass

        # Get lidar information
        lidar_extrinsics = self.get_current('extrinsics', depth_idx)
        lidar_points = self.get_current('point_cloud', depth_idx)
        world_points = (lidar_extrinsics * lidar_points).T

        # Create camera
        camera = PanoCamera(K[None], hw, Twc=Twc[None])
        world_points = torch.FloatTensor(world_points[None])

        # Generate depth maps
        depth, angle = generate_pano_proj_maps(camera, world_points, lidar_points)

        save_to_file(filename_depth, {'depth': depth, 'angle': angle})
        return depth, angle


    def __getitem__(self, idx):
        samples = super().__getitem__(idx)
        filename_chunk = self.get_filename(idx, 0).split('/')
        filename_chunk[-2] = os.path.join(PANO_CAMERA_NAME.upper(), self.pano_name)
        filename = '/'.join(filename_chunk)

        # lidar_pose = torch.FloatTensor(self.get_current('pose', self.depth_idx).matrix).inverse()

        # Extrinsics: A pose of sensor wrt the body frame. (but maybe corrupted? not sure yet.)
        # Pose: 4 x 4 transformation from sensor to world

        # From dgp/datasets/base_dataset.py:L1381
        # "extrinsics": Pose
        #   Camera extrinsics with respect to the vehicle frame, if available.
        # "pose": Pose
        #   Pose of sensor with respect to the world/global/local frame
        #   (reference frame that is initialized at start-time). (i.e. this
        #   provides the ego-pose in `pose_WC`).

        # TODO(soonminh): follow the same convention with OuroborosDataset
        # e.g. some entities such as intrinsics and depth should be dicts by context index
        params = PanoCamera.params_from_config(self.pano_cfg)
        hw = params['hw']
        K = params['K']
        # Twc = params['Twc'] @ lidar_pose
        # extrinsics = params['Twc']
        # Twc = extrinsics @ lidar_pose
        # Twc = extrinsics
        Twc = params['Twc']
        samples[PANO_CAMERA_NAME.lower()] = {
            'filename': {0: filename},
            'hw': hw,
            'intrinsics': {0: K},
            'Twc': Twc,
            # 'extrinsics': {0: extrinsics}
        }

        for i in range(self.num_cameras):
            camera = self.get_current('datum_name', i).lower()
            pose_to_pano = Twc @ torch.FloatTensor(samples[camera]['extrinsics'][0]).inverse()
            # pose_to_pano = Twc @ torch.FloatTensor(samples[camera]['pose'][0]).inverse() # (camera -> world) -> (world -> pano) == (camera -> pano)
            # pose_to_pano_orig = params['Twc'] @ torch.FloatTensor(samples[camera]['extrinsics'][0]).inverse()
            samples[camera]['pose_to_pano'] = {0: pose_to_pano}

        if self.with_depth:
            depth, angle = self.create_pano_proj_maps(filename, K, hw, Twc, self.depth_idx, self.depth_type)
            depth = resize_npy_preserve(depth, hw, expand_dims=False)
            angle = resize_npy_preserve(angle, hw, expand_dims=False)
            samples[PANO_CAMERA_NAME.lower()]['depth'] = {0: depth.astype(np.float32)[None]}
            samples[PANO_CAMERA_NAME.lower()]['angle'] = {0: angle.astype(np.float32)[None]}

        return samples


if __name__ == '__main__':
    """Generate RGB + PanoDepth image for debugging"""

    import cv2
    from PIL import Image
    from struct import pack, unpack

    from vidar.datasets.utils.transforms import get_transforms
    from vidar.utils.config import Config
    from vidar.utils.viz import viz_depth

    height, width = (384, 640)
    height_pano, width_pano = (256, 2048)
    params = {
        'name': 'PanoCamOuroboros',
        # 'path': '/data/datasets/DDAD_tiny/ddad_tiny_000071.json',
        'path': '/data/datasets/DDAD/ddad_train_val/ddad.json',
        'split': 'train',
        'context': [-1, 1],
        'labels': ['depth', 'pose'],
        'cameras': [1, 5, 6, 7, 8, 9],
        'depth_type': 'lidar',
        'repeat': 1,
        'pano_cam_config': Config(**{
            'name': 'panocam_150_z_-02_+02',
            'height': height_pano,
            'width': width_pano,
            'position_in_world': [0, 0, 1.5],
            'phi_range': [0, 6.2831853072],
            'rho': 1.0,
            'z_range': [-0.2, 0.2]}),
        'data_transform':
            get_transforms('train', Config(**{'resize': [height, width]})),
    }

    # Initialize dataset
    dataset = PanoCamOuroborosDataset(**params)
    print('# of frames: {}'.format(len(dataset)))

    # Get a sample
    data = dataset[0]

    scene, *_, timestamp = data['camera_01']['filename'][0].split('/')
    filename = f'{scene}_{timestamp}'
    panodepth = data['camera_pano']['depth']
    panodepth_viz = (viz_depth(panodepth[0][0], filter_zeros=False) * 255.0).astype(np.uint8)
    tensor_to_rgb_viz = lambda x: (x.permute(1, 2, 0) * 255.0).numpy().astype(np.uint8)
    rgbs = np.hstack(
        [tensor_to_rgb_viz(data[c]['rgb'][0]) for c in data.keys() if c.startswith('camera_0')]
    )

    rgbs = cv2.resize(rgbs, (panodepth_viz.shape[1], panodepth_viz.shape[0]))

    # From PanoDepth to ImageDepth
    panodepth_tensor = torch.FloatTensor(panodepth[0])[None]
    params = PanoCamera.params_from_config(params['pano_cam_config'].dict)
    camera = PanoCamera(params['K'][None], params['hw'], params['Twc'][None])

    # xyz_lidar = camera.reconstruct_depth_map(panodepth_tensor, to_world=True)
    xyz_lidar = camera.reconstruct_depth_map(panodepth_tensor, to_world=False)
    xyz_lidar = xyz_lidar.view(3, -1).numpy()
    rgb_lidar = np.zeros_like(xyz_lidar).T
    import pdb; pdb.set_trace()
    for camera in dataset.sensors:
        if not camera.startswith('camera'):
            continue
        print('RGB from {}'.format(data[camera]['filename'][0]))

        K = data[camera]['intrinsics'][0].numpy()
        # Twc = data[camera]['extrinsics'][0].numpy()
        # xyz_camera = Twc[:3, :3] @ xyz_lidar + Twc[:3, 3:]
        Tpc = np.linalg.inv(data[camera]['pose_to_pano'][0])
        xyz_camera = Tpc[:3, :3] @ xyz_lidar + Tpc[:3, 3:]
        ix, iy, iz = K @ xyz_camera
        ix, iy = ((ix / iz).astype(np.int16), (iy / iz).astype(np.int16))

        proj_on_image = np.logical_and.reduce([
            xyz_camera[2] > 0,
            ix >= 0, ix < width,
            iy >= 0, iy < height,
        ])

        image = data[camera]['rgb'][0].permute(1, 2, 0).numpy()
        rgb_lidar[proj_on_image] = image[iy[proj_on_image], ix[proj_on_image], :]

    xyz_lidar = xyz_lidar.T
    rgb_lidar = (rgb_lidar * 255.0).astype(np.uint8)

    rgb_lidar_image = rgb_lidar.reshape(height_pano, width_pano, 3)
    Image.fromarray(np.vstack([rgbs, panodepth_viz, rgb_lidar_image])).save(filename + '_frame.png')
    print('Save to {}'.format(filename + '_frame.png'))

    # Save to pcd/ply
    save_to_ply = False
    if save_to_ply:
        # preview on mac
        HEADER = (
            'ply',
            'format ascii 1.0',
            'element vertex {}'.format(len(xyz_lidar)),
            'property float x',
            'property float y',
            'property float z',
            'property uchar red',
            'property uchar green',
            'property uchar blue',
            'end_header',
        )
        suffix = '_point_cloud.ply'
        def write_func(f, x, y, z, r, g, b):
            f.write(('{:.4f} ' * 6 + '\n').format(x, y, z, r, g, b))
    else:
        # open3d
        HEADER = (
            '# .PCD v0.7 - Point Cloud Data file format',
            'VERSION 0.7',
            'FIELDS x y z rgb',
            'SIZE 4 4 4 4',
            'TYPE F F F F',
            'COUNT 1 1 1 1',
            'WIDTH {}'.format(len(xyz_lidar)),
            'HEIGHT 1',
            'VIEWPOINT 0 0 0 1 0 0 0',
            'POINTS {}'.format(len(xyz_lidar)),
            'DATA ascii',
        )
        suffix = '_point_cloud.pcd'
        def write_func(f, x, y, z, r, g, b):
            rgb = b | g << 8 | r << 16
            rgb = unpack('f', pack('i', rgb))[0]
            f.write(('{} ' * 3 + '{}\n').format(x, y, z, rgb))

    rgb_lidar = rgb_lidar.astype(np.uint32)
    print('Save to {}'.format(filename + suffix))
    with open(filename + suffix, 'w') as f:
        f.write('\n'.join(HEADER) + '\n')
        for (x, y, z), (r, g, b) in zip(xyz_lidar, rgb_lidar):
            write_func(f, x, y, z, r, g, b)