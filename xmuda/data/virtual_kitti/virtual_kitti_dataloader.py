import os.path as osp
import pickle
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T

from xmuda.data.utils.augmentation_3d import augment_and_scale_3d


class VirtualKITTIBase(Dataset):
    """Virtual Kitti dataset"""
    # https://github.com/VisualComputingInstitute/vkitti3D-dataset

    class_names = [
        "Terrain",
        "Tree",
        "Vegetation",
        "Building",
        "Road",
        "GuardRail",
        "TrafficSign",
        "TrafficLight",
        "Pole",
        "Misc",
        "Truck",
        "Car",
        "Van",
        "Don't care",
    ]

    # use those categories if merge_classes == True
    categories = {
        'vegetation_terrain': ['Terrain', 'Tree', 'Vegetation'],
        'building': ['Building'],
        'road': ['Road'],
        'object': ['TrafficSign', 'TrafficLight', 'Pole', 'Misc'],
        'truck': ['Truck'],
        'car': ['Car'],
        # 'ignore': ['Van', "Don't care"]
    }

    proj_matrix = np.array([[725, 0, 620.5], [0, 725, 187], [0, 0, 1]], dtype=np.float32)
    # weather_list = ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']

    def __init__(self,
                 split,
                 preprocess_dir,
                 merge_classes=False
                 ):

        self.split = split
        self.preprocess_dir = preprocess_dir

        print("Initialize Virtual Kitti dataloader")

        assert isinstance(split, tuple)
        print('Load', split)
        self.data = []
        for curr_split in split:
            with open(osp.join(self.preprocess_dir, curr_split + '.pkl'), 'rb') as f:
                self.data.extend(pickle.load(f))

        if merge_classes:
            self.label_mapping = -100 * np.ones(len(self.class_names), dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_names.index(class_name)] = cat_idx
            self.class_names = list(self.categories.keys())
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class VirtualKITTISCN(VirtualKITTIBase):
    def __init__(self,
                 split,
                 preprocess_dir,
                 virtual_kitti_dir='',
                 merge_classes=False,
                 scale=20,
                 full_scale=4096,
                 image_normalizer=None,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_y=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 downsample=(-1,),  # 3D augmentation
                 crop_size=tuple(),
                 bottom_crop=False,
                 rand_crop=tuple(),  # 2D augmentation
                 fliplr=0.0,  # 2D augmentation
                 color_jitter=None,  # 2D augmentation
                 random_weather=tuple()  # 2D augmentation
                 ):
        super().__init__(split,
                         preprocess_dir,
                         merge_classes=merge_classes)

        self.virtual_kitti_dir = virtual_kitti_dir

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_y = flip_y
        self.rot_z = rot_z
        self.transl = transl
        assert isinstance(downsample, tuple)
        if len(downsample) == 1:
            self.downsample = downsample[0]
        elif len(downsample) == 2:
            self.downsample = downsample
        else:
            NotImplementedError('Downsample must be either a tuple of (num_points,) or (min_points, max_points),'
                                'such that a different number of points can be sampled randomly for each example.')

        # image parameters
        self.image_normalizer = image_normalizer
        # 2D augmentation
        self.crop_size = crop_size
        if self.crop_size:
            assert bottom_crop != bool(rand_crop), 'Exactly one crop method needs to be active if crop size is provided!'
        else:
            assert not bottom_crop and not rand_crop, 'No crop size, but crop method is provided is provided!'
        self.bottom_crop = bottom_crop
        self.rand_crop = np.array(rand_crop)
        assert len(self.rand_crop) in [0, 4]

        self.random_weather = random_weather
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None

    def __getitem__(self, index):
        data_dict = self.data[index]

        points = data_dict['points'].copy()
        seg_label = data_dict['seg_labels'].astype(np.int64)

        # uniformly downsample point cloud without replacement
        num_points = self.downsample
        if isinstance(num_points, tuple):
            num_points = np.random.randint(low=num_points[0], high=num_points[1])
        if num_points > 0:
            assert num_points < len(points)
            choice = np.random.choice(len(points), size=num_points, replace=False)
            points = points[choice]
            seg_label = seg_label[choice]


        if self.label_mapping is not None:
            seg_label = self.label_mapping[seg_label]

        out_dict = {}

        keep_idx = np.ones(len(points), dtype=np.bool)
        # project points into image
        points_cam_coords = np.array([-1, -1, 1]) * points[:, [1, 2, 0]]
        points_img = (self.proj_matrix @ points_cam_coords.T).T
        points_img = points_img[:, :2] / np.expand_dims(points_img[:, 2], axis=1)  # scale 2D points
        # fliplr so that indexing is row, col and not col, row
        points_img = np.fliplr(points_img)

        # load image
        weather = 'clone'
        if self.random_weather:
            weather = self.random_weather[np.random.randint(len(self.random_weather))]
        img_path = osp.join(self.virtual_kitti_dir, 'vkitti_1.3.1_rgb', data_dict['scene_id'], weather,
                            data_dict['frame_id'] + '.png')
        image = Image.open(img_path)

        if self.crop_size:
            # self.crop_size is a tuple (crop_width, crop_height)
            valid_crop = False
            for _ in range(10):
                if self.bottom_crop:
                    # self.bottom_crop is a boolean
                    left = int(np.random.rand() * (image.size[0] + 1 - self.crop_size[0]))
                    right = left + self.crop_size[0]
                    top = image.size[1] - self.crop_size[1]
                    bottom = image.size[1]
                elif len(self.rand_crop) > 0:
                    # self.rand_crop is a tuple of floats in interval (0, 1):
                    # (min_crop_height, max_crop_height, min_crop_width, max_crop_width)
                    crop_height, crop_width = self.rand_crop[0::2] + \
                                              np.random.rand(2) * (self.rand_crop[1::2] - self.rand_crop[0::2])
                    top = np.random.rand() * (1 - crop_height) * image.size[1]
                    left = np.random.rand() * (1 - crop_width) * image.size[0]
                    bottom = top + crop_height * image.size[1]
                    right = left + crop_width * image.size[0]
                    top, left, bottom, right = int(top), int(left), int(bottom), int(right)

                # discard points outside of crop
                keep_idx = points_img[:, 0] >= top
                keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
                keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
                keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)

                if np.sum(keep_idx) > 100:
                    valid_crop = True
                    break

            if valid_crop:
                # crop image
                image = image.crop((left, top, right, bottom))
                points_img = points_img[keep_idx]
                points_img[:, 0] -= top
                points_img[:, 1] -= left

                # update point cloud
                points = points[keep_idx]
                if seg_label is not None:
                    seg_label = seg_label[keep_idx]

                if len(self.rand_crop) > 0:
                    # scale image points
                    points_img[:, 0] = float(self.crop_size[1]) / image.size[1] * np.floor(points_img[:, 0])
                    points_img[:, 1] = float(self.crop_size[0]) / image.size[0] * np.floor(points_img[:, 1])

                    # resize image (only during random crop, never during test)
                    image = image.resize(self.crop_size, Image.BILINEAR)
            else:
                print('No valid crop found for image', data_dict['camera_path'])

        img_indices = points_img.astype(np.int64)

        assert np.all(img_indices[:, 0] >= 0)
        assert np.all(img_indices[:, 1] >= 0)
        assert np.all(img_indices[:, 0] < image.size[1])
        assert np.all(img_indices[:, 1] < image.size[0])

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.
        # 2D augmentation
        if np.random.rand() < self.fliplr:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        out_dict['img'] = np.moveaxis(image, -1, 0)
        out_dict['img_indices'] = img_indices

        # 3D data augmentation and scaling from points to voxel indices
        # Kitti lidar coordinates: x (front), y (left), z (up)
        coords = augment_and_scale_3d(points, self.scale, self.full_scale, noisy_rot=self.noisy_rot,
                                      flip_y=self.flip_y, rot_z=self.rot_z, transl=self.transl)

        # cast to integer
        coords = coords.astype(np.int64)

        # only use voxels inside receptive field
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)

        out_dict['coords'] = coords[idxs]
        out_dict['feats'] = np.ones([len(idxs), 1], np.float32)  # simply use 1 as feature
        out_dict['seg_label'] = seg_label[idxs]
        out_dict['img_indices'] = out_dict['img_indices'][idxs]

        return out_dict


def test_VirtualKITTISCN():
    from xmuda.data.utils.visualize import draw_points_image_labels, draw_bird_eye_view
    preprocess_dir = '/datasets_local/datasets_mjaritz/virtual_kitti_preprocess/preprocess'
    virtual_kitti_dir = '/datasets_local/datasets_mjaritz/virtual_kitti_preprocess'
    split = ('mini',)
    dataset = VirtualKITTISCN(split=split,
                              preprocess_dir=preprocess_dir,
                              virtual_kitti_dir=virtual_kitti_dir,
                              merge_classes=True,
                              noisy_rot=0.1,
                              flip_y=0.5,
                              rot_z=2*np.pi,
                              transl=True,
                              downsample=(10000,),
                              crop_size=(480, 302),
                              bottom_crop=True,
                              # rand_crop=(0.7, 1.0, 0.3, 0.5),
                              fliplr=0.5,
                              color_jitter=(0.4, 0.4, 0.4),
                              # random_weather=('clone', 'morning', 'overcast', 'sunset')
                              )
    # for i in [10, 20, 30, 40, 50, 60]:
    for i in 5 * [0]:
        data = dataset[i]
        coords = data['coords']
        seg_label = data['seg_label']
        img = np.moveaxis(data['img'], 0, 2)
        img_indices = data['img_indices']
        draw_points_image_labels(img, img_indices, seg_label, color_palette_type='VirtualKITTI', point_size=3)
        draw_bird_eye_view(coords)
        print(len(coords))


def compute_class_weights():
    preprocess_dir = '/datasets_local/datasets_mjaritz/virtual_kitti_preprocess/preprocess'
    split = ('train',)
    dataset = VirtualKITTIBase(split,
                               preprocess_dir,
                               merge_classes=True
                               )
    # compute points per class over whole dataset
    num_classes = len(dataset.class_names)
    points_per_class = np.zeros(num_classes, int)
    for i, data in enumerate(dataset.data):
        print('{}/{}'.format(i, len(dataset)))
        labels = dataset.label_mapping[data['seg_labels']]
        points_per_class += np.bincount(labels[labels != -100], minlength=num_classes)

    # compute log smoothed class weights
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    print('log smoothed class weights: ', class_weights / class_weights.min())


if __name__ == '__main__':
    # test_VirtualKITTISCN()
    compute_class_weights()
