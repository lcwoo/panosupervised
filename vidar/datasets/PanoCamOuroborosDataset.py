import numpy as np
import os
import torch

from vidar.datasets.augmentations.resize import resize_npy_preserve
from vidar.geometry.camera_pano import PanoCamera
from vidar.geometry.camera import Camera
from vidar.datasets.OuroborosDataset import load_from_file, save_to_file, generate_proj_maps
from vidar.datasets.OuroborosDataset import OuroborosDataset
from vidar.datasets.utils.misc import stack_sample


PANO_CAMERA_NAME = 'camera_pano'


def generate_pano_proj_maps(camera, Xw, Xl,Twc):
    # Project the points
    uv_tensor, rho_tensor = camera.project_points_with_cam1(Xw,Twc, normalize=False, return_z=True)
    uv = uv_tensor[0].long()  # Convert to long for indexing
    rho = rho_tensor[0]

    # Create an empty image to overlay
    H, W = camera.hw
    proj_depth = torch.zeros((H, W), dtype=torch.float32).to(uv.device)
    
    in_view = (uv >= 0).all(dim=1) & (uv[:, 0] < W) & (uv[:, 1] < H) & (rho > 0)
    uv = uv[in_view]
    rho = rho[in_view]

    # Sort by distance to pick closest one if multiple LiDAR points are projected onto a single pixel
    order = torch.argsort(rho, descending=True)
    uv = uv[order]
    rho = rho[order]
    
    # Direct indexing on tensors to fill the image
    proj_depth[uv[:, 1], uv[:, 0]] = rho

    # Calculate yaw angle in LiDAR coordinate
    xx = Xl[in_view][:, 0]
    yy = Xl[in_view][:, 1]
    yaw = torch.atan2(yy, xx + 1e-6)

    # Reverse yaw to make it clockwise and add pi to start from backward

    proj_angle = torch.zeros((H, W), dtype=torch.float32, device=uv.device)
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
        

    def panodepth_to_points(self, distance):
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
        params = PanoCamera.params_from_config(self.pano_cfg)
        hw = params['hw']
        K = params['K']

        Twc = params['Twc']
        h, w = hw
        pcl = PanoCamera(K[None], hw, Twc=Twc[None]).reconstruct_depth_map(
            distance * torch.ones([1, 1, h, w]), to_world=True)
        return pcl.numpy().reshape(3, -1).T


    def create_pano_rays(self, distance=1.0):
        image_shape = PanoCamera.params_from_config(self.pano_cfg)['hw']
        pcl = self.panodepth_to_points(distance)

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

        rays = np.stack([theta, phi], axis=0).reshape(2, *image_shape)
        return rays


        ### Calculate polar/azimuth angles in LiDAR coordinate
        # [LiDAR coordinate convention] https://en.wikipedia.org/wiki/Spherical_coordinate_system
        #   (X, Y, Z) = (N, W, U), where northing (N), westing(W), and upwardness (U)
        #   polar angle (theta): measured from a fixed zenith direction
        #   azimuth angle (phi): measured from negative X-axis (southing, S) on a reference (XY-) plane

        image_shape = (self.pano_cfg['height'], self.pano_cfg['width'])

        # TODO(soonminh): change to spherical coordinate
        rho = self.pano_cfg['rho']
        theta_range = [np.arctan2(abs(z), rho) for z in self.pano_cfg['z_range']]
        theta_range[0] = np.pi / 2 - theta_range[0]
        theta_range[1] += np.pi / 2

        theta = np.linspace(*theta_range, num=image_shape[0])
        phi = np.linspace(0, 2 * np.pi + 1e-6, num=image_shape[1])

        theta, phi = np.meshgrid(theta, phi, indexing='ij')
        rays = np.stack([theta, phi], axis=0)
        return rays
    
    def create_pano_mask(self, sample, K, hw, Twc, depth_idx, depth_type):
        """
        Creates a panoramic mask by transforming and aggregating masks from multiple cameras.

        Parameters
        ----------
        samples : dict
            Dictionary containing sample data from multiple cameras.
        K : np.Array
            Intrinsic matrix of the panoramic camera.
        hw : tuple
            Height and width of the panoramic camera.
        Twc : np.Array
            Extrinsic matrix of the panoramic camera.

        Returns
        -------
        pano_mask :
            Aggregated panoramic mask.
        """
        pano_masks = []
        camera_names = [k for k in sample.keys() if k.startswith('camera_0')]
        K_pano = sample[PANO_CAMERA_NAME]['intrinsics'][0].float()

        for cam_name in camera_names:
            
            pose_to_pano = sample[cam_name]['pose_to_pano'][0]
            K = sample[cam_name]['intrinsics'][0]
            mask_i = sample[cam_name]['mask']
            _, height, width = mask_i.size()
            
            coords = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing='xy'), dim=2).float()

            mask_values = mask_i.squeeze(0).unsqueeze(-1) # [1, 384, 640] -> [384, 640, 1]

            coords = coords.transpose(0, 1)  # 이제 coords의 크기는 [640, 384, 2]가 됩니다.

            # 최종 좌표 생성: (x, y, mask_value)
            final_coords = torch.cat([coords, mask_values], dim=2)

            
            K_inv = torch.inverse(K.float())
            camera_mask = torch.einsum('ij,klj->kli', K_inv, final_coords) 
            # Create a mask for the camera
            homogeneous_camera_mask = torch.cat([camera_mask, torch.ones(384, 640, 1)], dim=2)  # 마지막 차원에 1 추가
            flat_homogeneous_camera_mask = homogeneous_camera_mask.reshape(-1, 4)
            pano_coords = torch.mm(flat_homogeneous_camera_mask, pose_to_pano.transpose(0, 1))  #c 행렬 곱

            # Camera to Image 플레인 투영
            pano_coords = pano_coords[:, :3]
            projected_coords = torch.mm(pano_coords, K_pano.transpose(0, 1))

            pano_masks.append(projected_coords)

        # Stack all masks
        
        pano_masks = torch.stack(pano_masks, dim=0)

        return pano_mask

    
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
        pose = self.get_current('extrinsics', depth_idx)
        transformation_matrix = torch.tensor(pose.matrix, dtype=torch.float32).to(torch.device('cuda'))

        lidar_points_tensor = torch.tensor(self.get_current('point_cloud', depth_idx), dtype=torch.float32).to(torch.device('cuda'))
        homogeneous_points = torch.cat((lidar_points_tensor, torch.ones((lidar_points_tensor.shape[0], 1), dtype=torch.float32,device=torch.device('cuda'))), dim=1)
        # world_points = torch.matmul(transformation_matrix, homogeneous_points.T)
        world_points = torch.matmul(transformation_matrix, homogeneous_points.T).T[:, :3].unsqueeze(0).transpose(1, 2)
        # camera_points = torch.matmul(Twc_1, world_points).T[:, :3].unsqueeze(0).transpose(1, 2)
        # # HACK: (chungwoo): 나중 코드 작성 편의를 위해 camera 좌표계에 있는 camera_points를 월드 좌표계의 순서대로 바꿈 -x-> y, z -> x, x y
        # T = torch.tensor([
        #     [0, 0, 1],
        #     [-1, 0, 0],
        #     [0, -1, 0]
        # ], dtype=torch.float32)

        # camera_points_transformed = torch.matmul(T,camera_points)

        # Create camera
        camera = PanoCamera(K[None], hw, Twc=Twc[None])

        # Generate depth maps
        depth, angle = generate_pano_proj_maps(camera, world_points, lidar_points_tensor, Twc)
        # depth, angle = generate_pano_proj_maps(camera, camera_points, lidar_points_tensor)
        # depth, angle = generate_pano_proj_maps(camera, camera_points_transformed, lidar_points_tensor)
        
        depth = depth.detach().cpu().numpy()
        angle = angle.detach().cpu().numpy()

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

        # import ipdb; ipdb.set_trace()
        hw = params['hw']
        K = params['K']
        # Twc = params['Twc'] @ lidar_pose
        # extrinsics = params['Twc']
        # Twc = extrinsics @ lidar_pose
        # Twc = extrinsics
        Twc = torch.FloatTensor(samples['camera_01']['extrinsics'][0])
        samples[PANO_CAMERA_NAME.lower()] = {
            'filename': {0: filename},
            'hw': hw,
            'intrinsics': {0: K},
            'Twc': Twc,
            # 'extrinsics': {0: extrinsics}
        }

        # Rays
        if self.with_rays:
            rays = self.create_pano_rays()
            embedding = np.stack([
                np.sin(rays[0]),
                np.cos(rays[0]),
                np.sin(rays[1]),
                np.cos(rays[1]),
            ], axis=0)
            samples[PANO_CAMERA_NAME.lower()].update({
                'rays': {0: rays},
                'rays_embedding': {0: embedding},
            })

        for i in range(self.num_cameras):
            camera = self.get_current('datum_name', i).lower()
            pose_to_pano = Twc @ (torch.FloatTensor(samples[camera]['extrinsics'][0]).inverse())
            
            # pose_to_pano = Twc @ torch.FloatTensor(samples[camera]['pose'][0]).inverse() # (camera -> world) -> (world -> pano) == (camera -> pano)
            # pose_to_pano_orig = params['Twc'] @ torch.FloatTensor(samples[camera]['extrinsics'][0]).inverse()
            samples[camera]['pose_to_pano'] = {0: pose_to_pano}
        #TODO: make pano_mask
        # panoramic_mask = self.create_pano_mask(samples, K, hw, Twc, self.depth_idx, self.depth_type)
        # samples[PANO_CAMERA_NAME.lower()]['mask'] = {0: panoramic_mask}
        
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
        'path': '/data/datasets/DDAD/ddad_train_val/ddad_overfit_000071.json',
        # 'path': '/data/datasets/DDAD/ddad_train_val/ddad.json',
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
            'position_in_world': [0, 0, 0],
            'phi_range': [0, 6.2831853072],
            'rho': 1.0,
            'z_range': [-0.2, 0.2]}),
        'data_transform':
            get_transforms('train', Config(**{'resize': [height, width]})),
    }

    # Initialize dataset
    dataset = PanoCamOuroborosDataset(**params)
    print('# of frames: {}'.format(len(dataset)))
    # '000071/{}/CAMERA_01/1568648972095306'
    # Get a sample
    data = dataset[8]

    scene, *_, timestamp = data['camera_01']['filename'][0].split('/')
    filename = f'{scene}_{1568648972095306}'
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

        image = data[camera]['rgb'][0].permute(2, 1, 0).numpy()
        rgb_lidar[proj_on_image] = image[ix[proj_on_image], iy[proj_on_image], :]

    xyz_lidar = xyz_lidar.T
    rgb_lidar = (rgb_lidar * 255.0).astype(np.uint8)

    rgb_lidar_image = rgb_lidar.reshape(height_pano, width_pano, 3)
    Image.fromarray(np.vstack([rgbs, panodepth_viz, rgb_lidar_image])).save(filename + '_frame1.png')
    print('Save to {}'.format(filename + '_frame1.png'))

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