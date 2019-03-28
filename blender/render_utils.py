from config import cfg
from base_utils import PoseTransformer, read_pose, read_pickle, save_pickle
import os
import numpy as np
from transforms3d.quaternions import mat2quat
import glob
from PIL import Image
from scipy import stats
import OpenEXR
import Imath
from multiprocessing.dummy import Pool
import struct
import scipy.io as sio


class DataStatistics(object):
    # world_to_camera_pose = np.array([[-1.19209304e-07,   1.00000000e+00,  -2.98023188e-08, 1.19209304e-07],
    #                                  [-8.94069672e-08,   2.22044605e-16,  -1.00000000e+00, 8.94069672e-08],
    #                                  [-1.00000000e+00,  -8.94069672e-08,   1.19209304e-07, 1.00000000e+00]])
    world_to_camera_pose = np.array([[-1.00000024e+00,  -8.74227979e-08,  -5.02429621e-15, 8.74227979e-08],
                                     [5.02429621e-15,   1.34358856e-07,  -1.00000012e+00, -1.34358856e-07],
                                     [8.74227979e-08,  -1.00000012e+00,   1.34358856e-07, 1.00000012e+00]])

    def __init__(self, class_type):
        self.class_type = class_type
        self.mask_path = os.path.join(cfg.LINEMOD,'{}/mask/*.png'.format(class_type))
        self.dir_path = os.path.join(cfg.LINEMOD_ORIG,'{}/data'.format(class_type))

        dataset_pose_dir_path = os.path.join(cfg.DATA_DIR, 'dataset_poses')
        os.system('mkdir -p {}'.format(dataset_pose_dir_path))
        self.dataset_poses_path = os.path.join(dataset_pose_dir_path, '{}_poses.npy'.format(class_type))
        blender_pose_dir_path = os.path.join(cfg.DATA_DIR, 'blender_poses')
        os.system('mkdir -p {}'.format(blender_pose_dir_path))
        self.blender_poses_path = os.path.join(blender_pose_dir_path, '{}_poses.npy'.format(class_type))
        os.system('mkdir -p {}'.format(blender_pose_dir_path))

        self.pose_transformer = PoseTransformer(class_type)

    def get_proper_crop_size(self):
        mask_paths = glob.glob(self.mask_path)
        widths = []
        heights = []

        for mask_path in mask_paths:
            mask = Image.open(mask_path).convert('1')
            mask = np.array(mask).astype(np.int32)
            row_col = np.argwhere(mask == 1)
            min_row, max_row = np.min(row_col[:, 0]), np.max(row_col[:, 0])
            min_col, max_col = np.min(row_col[:, 1]), np.max(row_col[:, 1])
            width = max_col - min_col
            height = max_row - min_row
            widths.append(width)
            heights.append(height)

        widths = np.array(widths)
        heights = np.array(heights)
        print('min width: {}, max width: {}'.format(np.min(widths), np.max(widths)))
        print('min height: {}, max height: {}'.format(np.min(heights), np.max(heights)))

    def get_quat_translation(self, object_to_camera_pose):
        object_to_camera_pose = np.append(object_to_camera_pose, [[0, 0, 0, 1]], axis=0)
        world_to_camera_pose = np.append(self.world_to_camera_pose, [[0, 0, 0, 1]], axis=0)
        object_to_world_pose = np.dot(np.linalg.inv(world_to_camera_pose), object_to_camera_pose)
        quat = mat2quat(object_to_world_pose[:3, :3])
        translation = object_to_world_pose[:3, 3]
        return quat, translation

    def get_dataset_poses(self):
        if os.path.exists(self.dataset_poses_path):
            poses = np.load(self.dataset_poses_path)
            return poses[:, :3], poses[:, 3:]

        eulers = []
        translations = []
        train_set = np.loadtxt(os.path.join(cfg.LINEMOD, '{}/training_range.txt'.format(self.class_type)),np.int32)
        for idx in train_set:
            rot_path = os.path.join(self.dir_path, 'rot{}.rot'.format(idx))
            tra_path = os.path.join(self.dir_path, 'tra{}.tra'.format(idx))
            pose = read_pose(rot_path, tra_path)
            euler = self.pose_transformer.orig_pose_to_blender_euler(pose)
            eulers.append(euler)
            translations.append(pose[:, 3])

        eulers = np.array(eulers)
        translations = np.array(translations)
        np.save(self.dataset_poses_path, np.concatenate([eulers, translations], axis=-1))

        return eulers, translations

    def sample_sphere(self, num_samples):
        """ sample angles from the sphere
        reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
        """
        flat_objects = ['037_scissors', '051_large_clamp', '052_extra_large_clamp']
        if self.class_type in flat_objects:
            begin_elevation = 30
        else:
            begin_elevation = 0
        ratio = (begin_elevation + 90) / 180
        num_points = int(num_samples // (1 - ratio))
        phi = (np.sqrt(5) - 1.0) / 2.
        azimuths = []
        elevations = []
        for n in range(num_points - num_samples, num_points):
            z = 2. * n / num_points - 1.
            azimuths.append(np.rad2deg(2 * np.pi * n * phi % (2 * np.pi)))
            elevations.append(np.rad2deg(np.arcsin(z)))
        return np.array(azimuths), np.array(elevations)

    def sample_poses(self):
        eulers, translations = self.get_dataset_poses()
        num_samples = cfg.NUM_SYN
        azimuths, elevations = self.sample_sphere(num_samples)
        euler_sampler = stats.gaussian_kde(eulers.T)
        eulers = euler_sampler.resample(num_samples).T
        eulers[:, 0] = azimuths
        eulers[:, 1] = elevations
        translation_sampler = stats.gaussian_kde(translations.T)
        translations = translation_sampler.resample(num_samples).T
        np.save(self.blender_poses_path, np.concatenate([eulers, translations], axis=-1))


class YCBDataStatistics(DataStatistics):
    def __init__(self, class_type):
        super(YCBDataStatistics, self).__init__(class_type)
        self.dir_path = os.path.join(cfg.LINEMOD_ORIG, '{}/data'.format(class_type))
        self.class_types = np.loadtxt(os.path.join(cfg.YCB, 'image_sets/classes.txt'), dtype=np.str)
        self.class_types = np.insert(self.class_types, 0, 'background')
        self.train_set = np.loadtxt(os.path.join(cfg.YCB, 'image_sets/train.txt'), dtype=np.str)
        self.meta_pattern = os.path.join(cfg.YCB, 'data/{}-meta.mat')
        self.dataset_poses_pattern = os.path.join(cfg.DATA_DIR, 'dataset_poses/{}_poses.npy')

    def get_dataset_poses(self):
        if os.path.exists(self.dataset_poses_path):
            poses = np.load(self.dataset_poses_pattern.format(self.class_type))
            return poses[:, :3], poses[:, 3:]

        dataset_poses = {}
        for i in self.train_set:
            meta_path = self.meta_pattern.format(i)
            meta = sio.loadmat(meta_path)
            classes = meta['cls_indexes'].ravel()
            poses = meta['poses']
            for idx, cls_idx in enumerate(classes):
                cls_poses = dataset_poses.setdefault(self.class_types[cls_idx], [[], []])
                pose = poses[..., idx]
                euler = self.pose_transformer.blender_pose_to_blender_euler(pose)
                cls_poses[0].append(euler)
                cls_poses[1].append(pose[:, 3])

        for class_type, cls_poses in dataset_poses.items():
            np.save(self.dataset_poses_pattern.format(class_type), np.concatenate(cls_poses, axis=-1))

        cls_poses = dataset_poses[self.class_type]
        eulers = np.array(cls_poses[0])
        translations = np.array(cls_poses[1])

        return eulers, translations


class Renderer(object):
    intrinsic_matrix = {
        'linemod': np.array([[572.4114, 0., 325.2611],
                              [0., 573.57043, 242.04899],
                              [0., 0., 1.]]),
        # 'blender': np.array([[280.0, 0.0, 128.0],
        #                      [0.0, 280.0, 128.0],
        #                      [0.0, 0.0, 1.0]]),
        'blender': np.array([[700.,    0.,  320.],
                             [0.,  700.,  240.],
                             [0.,    0.,    1.]])
    }

    def __init__(self, class_type):
        self.class_type = class_type
        self.bg_imgs_path = os.path.join(cfg.DATA_DIR, 'bg_imgs.npy')
        self.poses_path = os.path.join(cfg.DATA_DIR, 'blender_poses', '{}_poses.npy').format(class_type)
        self.output_dir_path = os.path.join(cfg.LINEMOD,'renders/{}').format(class_type)
        self.blender_path = cfg.BLENDER_PATH
        self.blank_blend = os.path.join(cfg.DATA_DIR, 'blank.blend')
        self.py_path = os.path.join(cfg.BLENDER_DIR, 'render_backend.py')
        self.obj_path = os.path.join(cfg.LINEMOD,'{}/{}.ply').format(class_type, class_type)
        self.plane_height_path = os.path.join(cfg.DATA_DIR, 'plane_height.pkl')

    def get_bg_imgs(self):
        if os.path.exists(self.bg_imgs_path):
            return

        img_paths = glob.glob(os.path.join(cfg.SUN, 'JPEGImages/*'))
        bg_imgs = []

        for img_path in img_paths:
            img = Image.open(img_path)
            row, col = img.size
            if row > 500 and col > 500:
                bg_imgs.append(img_path)

        np.save(self.bg_imgs_path, bg_imgs)

    def project_model(self, model_3d, pose, camera_type):
        camera_model_2d = np.dot(model_3d, pose[:, :3].T) + pose[:, 3]
        camera_model_2d = np.dot(camera_model_2d, self.intrinsic_matrix[camera_type].T)
        return camera_model_2d[:, :2] / camera_model_2d[:, 2:]

    @staticmethod
    def exr_to_png(exr_path):
        depth_path = exr_path.replace('.png0001.exr', '.png')
        exr_image = OpenEXR.InputFile(exr_path)
        dw = exr_image.header()['dataWindow']
        (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        def read_exr(s, width, height):
            mat = np.fromstring(s, dtype=np.float32)
            mat = mat.reshape(height, width)
            return mat

        dmap, _, _ = [read_exr(s, width, height) for s in exr_image.channels('BGR', Imath.PixelType(Imath.PixelType.FLOAT))]
        dmap = Image.fromarray((dmap != 1).astype(np.int32))
        dmap.save(depth_path)
        exr_image.close()
        os.system('rm {}'.format(exr_path))

    def sample_poses(self):
        statistician = DataStatistics(self.class_type)
        statistician.sample_poses()

    def get_plane_height(self):
        if os.path.exists(self.plane_height_path):
            plane_height = read_pickle(self.plane_height_path)
        else:
            plane_height = {}

        if self.class_type in plane_height:
            return plane_height[self.class_type]
        else:
            pose_transformer = PoseTransformer(self.class_type)
            model = pose_transformer.get_blender_model()
            height = np.min(model[:, -1])
            plane_height[self.class_type] = height
            save_pickle(plane_height, self.plane_height_path)
            return height

    def run(self):
        """ Render images
        1. prepare background images
        2. sample poses from the pose distribution of training data
        3. call the blender to render images
        """
        self.get_bg_imgs()
        self.sample_poses()

        if not os.path.exists(self.output_dir_path):
            os.makedirs(self.output_dir_path)

        os.system('{} {} --background --python {} -- --input {} --output_dir {} --bg_imgs {} --poses_path {}'.
                  format(self.blender_path, self.blank_blend, self.py_path, self.obj_path,
                         self.output_dir_path, self.bg_imgs_path, self.poses_path))
        depth_paths = glob.glob(os.path.join(self.output_dir_path, '*.exr'))
        for depth_path in depth_paths:
            self.exr_to_png(depth_path)

    @staticmethod
    def multi_thread_render():
        # objects = ['ape', 'benchvise', 'bowl', 'can', 'cat', 'cup', 'driller', 'duck',
        #            'glue', 'holepuncher', 'iron', 'lamp', 'phone', 'cam', 'eggbox']
        objects = ['lamp', 'phone']

        def render(class_type):
            renderer = Renderer(class_type)
            renderer.run()

        with Pool(processes=2) as pool:
            pool.map(render, objects)


class YCBRenderer(Renderer):
    def __init__(self, class_type):
        super(YCBRenderer, self).__init__(class_type)
        self.output_dir_path = os.path.join(cfg.YCB, 'renders/{}').format(class_type)
        self.blank_blend = os.path.join(cfg.DATA_DIR, 'blank.blend')
        self.obj_path = os.path.join(cfg.YCB, 'models', class_type, 'textured.obj')
        self.class_types = np.loadtxt(os.path.join(cfg.YCB, 'image_sets/classes.txt'), dtype=np.str)
        self.class_types = np.insert(self.class_types, 0, 'background')

    def sample_poses(self):
        statistician = YCBDataStatistics(self.class_type)
        statistician.sample_poses()

    @staticmethod
    def multi_thread_render():
        objects = ['003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle']

        def render(class_type):
            renderer = YCBRenderer(class_type)
            renderer.run()

        with Pool(processes=2) as pool:
            pool.map(render, objects)


class MultiRenderer(Renderer):
    class_types = ['ape', 'benchvise', 'can', 'cat', 'driller', 'duck', 'glue',
                   'holepuncher', 'iron', 'lamp', 'phone', 'cam', 'eggbox']

    def __init__(self):
        super(MultiRenderer, self).__init__('')
        self.poses_path = os.path.join(cfg.DATA_DIR, '{}_poses.npy')
        self.output_dir_path = '/home/pengsida/Datasets/LINEMOD/renders/all_objects'

    def sample_poses(self):
        for class_type in self.class_types:
            statistician = DataStatistics(class_type)
            statistician.sample_poses()

    def run(self):
        """ Render images
        1. prepare background images
        2. sample poses from the pose distribution of training data
        3. call the blender to render images
        """
        self.get_bg_imgs()
        self.sample_poses()

        os.system('{} {} --background --python {} -- --input {} --output_dir {} --use_cycles True --bg_imgs {} --poses_path {}'.
                  format(self.blender_path, self.blank_blend, self.py_path, self.obj_path, self.output_dir_path, self.bg_imgs_path, self.poses_path))
        depth_paths = glob.glob(os.path.join(self.output_dir_path, '*.exr'))
        for depth_path in depth_paths:
            self.exr_to_png(depth_path)

