import numpy as np
import os
import torch

from vidar.datasets.augmentations.resize import resize_npy_preserve
from vidar.geometry.camera_pano import PanoCamera
from vidar.datasets.OuroborosDataset import load_from_file, save_to_file, generate_proj_maps
from vidar.datasets.OuroborosDataset import OuroborosDataset
from vidar.datasets.utils.misc import stack_sample


PANO_CAMERA_NAME = 'camera_pano'


def generate_pano_proj_maps(camera, Xw):
    """Render pointcloud on pano image.

    Parameters
    ----------
    camera: PanoCamera
        Camera object with appropriately set extrinsics wrt world.
    Xw: np.Array
        3D point cloud (x, y, z) in the world coordinate. [N,3]

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

    # Return projected maps
    return proj_depth


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

        # TODO(soonminh): cleaning up code
        params = PanoCamera.params_from_config(pano_cfg.dict)
        self.pano_K = params['K']
        self.pano_hw = params['hw']
        self.pano_Twc = params['Twc']
        self.pano_Tcw = self.pano_Twc.inverse()
        self.pano_cam = PanoCamera(self.pano_K[None], self.pano_hw, Twc=self.pano_Twc[None])

        self.pano_K = self.pano_K.numpy()
        self.pano_Twc = self.pano_Twc.numpy()
        self.pano_Tcw = self.pano_Tcw.numpy()

    def create_pano_proj_maps(self, filename, depth_idx, depth_type):
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
            depth = load_from_file(filename_depth, 'depth')
            return depth
        except:
            pass

        # Get lidar information
        lidar_extrinsics = self.get_current('extrinsics', depth_idx)
        lidar_points = self.get_current('point_cloud', depth_idx)
        world_points = (lidar_extrinsics * lidar_points).T

        # Create camera
        camera = self.pano_cam
        world_points = torch.FloatTensor(world_points[None])

        # Generate depth maps
        depth = generate_pano_proj_maps(camera, world_points)

        save_to_file(filename_depth, 'depth', depth)
        return depth

    def __getitem__(self, idx):
        samples = super().__getitem__(idx)
        filename_chunk = self.get_filename(idx, 0).split('/')
        filename_chunk[-2] = os.path.join(PANO_CAMERA_NAME.lower(), self.pano_name)
        filename = '/'.join(filename_chunk)

        # TODO(soonminh): follow the same convention with OuroborosDataset
        # e.g. some entities such as intrinsics and depth should be dicts by context index
        samples[PANO_CAMERA_NAME.lower()] = {
            'filename': filename,
            'hw': self.pano_hw,
            'intrinsics': self.pano_K,
            'Twc': self.pano_Twc,
        }

        for i in range(self.num_cameras):
            camera = self.get_current('datum_name', i).lower()
            pose_to_pano = (torch.FloatTensor(samples[camera]['extrinsics'][0]) @ self.pano_Tcw).inverse()
            samples[camera]['pose_to_pano'] = {0: pose_to_pano.numpy()}

        if self.with_depth:
            depth = self.create_pano_proj_maps(filename, self.depth_idx, self.depth_type)
            depth = resize_npy_preserve(depth, self.pano_hw, expand_dims=False)
            samples[PANO_CAMERA_NAME.lower()]['depth'] = depth.astype(np.float32)[None]

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
        'path': '/data/datasets/DDAD_tiny/ddad_tiny_000071.json',
        'split': 'train',
        'context': [-1, 1],
        'labels': ['depth', 'pose'],
        'cameras': [1, 5, 6, 7, 8, 9],
        'depth_type': 'lidar',
        'repeat': 1,
        'pano_cam_config': Config(**{
            'name': 'standard',
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

    panodepth_viz = (viz_depth(panodepth[0], filter_zeros=False) * 255.0).astype(np.uint8)
    tensor_to_rgb_viz = lambda x: (x.permute(1, 2, 0) * 255.0).numpy().astype(np.uint8)
    rgbs = np.hstack(
        [tensor_to_rgb_viz(data[c]['rgb'][0]) for c in data.keys() if c.startswith('camera_0')]
    )

    rgbs = cv2.resize(rgbs, (panodepth_viz.shape[1], panodepth_viz.shape[0]))

    # From PanoDepth to ImageDepth
    panodepth_tensor = torch.FloatTensor(panodepth)[None]
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