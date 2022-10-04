import os
import os.path as osp
import numpy as np
import pickle
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from xmuda.data.virtual_kitti import splits

# prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')


class DummyDataset(Dataset):
    def __init__(self, root_dir, scenes):
        self.root_dir = root_dir
        self.data = []
        self.glob_frames(scenes)

    def glob_frames(self, scenes):
        for scene in scenes:
            glob_path = osp.join(self.root_dir, 'vkitti3d_npy', scene, '*.npy')
            lidar_paths = sorted(glob.glob(glob_path))

            for lidar_path in lidar_paths:
                if not osp.exists(lidar_path):
                    raise IOError('File not found {}'.format(lidar_path))
                basename = osp.basename(lidar_path)
                frame_id = osp.splitext(basename)[0]
                assert frame_id.isdigit()
                data = {
                    'lidar_path': lidar_path,
                    'scene_id': scene,
                    'frame_id': frame_id
                }
                self.data.append(data)

    def __getitem__(self, index):
        data_dict = self.data[index].copy()
        point_cloud = np.load(data_dict['lidar_path'])
        points = point_cloud[:, :3].astype(np.float32)
        labels = point_cloud[:, 6].astype(np.uint8)

        data_dict['seg_label'] = labels
        data_dict['points'] = points

        return data_dict

    def __len__(self):
        return len(self.data)


def preprocess(split_name, root_dir, out_dir):
    pkl_data = []
    split = getattr(splits, split_name)

    dataloader = DataLoader(DummyDataset(root_dir, split), num_workers=10)

    num_skips = 0
    for i, data_dict in enumerate(dataloader):
        # data error leads to returning empty dict
        if not data_dict:
            print('empty dict, continue')
            num_skips += 1
            continue
        for k, v in data_dict.items():
            data_dict[k] = v[0]
        print('{}/{} {}'.format(i, len(dataloader), data_dict['lidar_path']))

        # convert to relative path
        lidar_path = data_dict['lidar_path'].replace(root_dir + '/', '')

        # append data
        out_dict = {
            'points': data_dict['points'].numpy(),
            'seg_labels': data_dict['seg_label'].numpy(),
            'lidar_path': lidar_path,
            'scene_id': data_dict['scene_id'],
            'frame_id': data_dict['frame_id']
        }
        pkl_data.append(out_dict)

    print('Skipped {} files'.format(num_skips))

    # save to pickle file
    save_dir = osp.join(out_dir, 'preprocess')
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, '{}.pkl'.format(split_name))
    with open(save_path, 'wb') as f:
        pickle.dump(pkl_data, f)
        print('Wrote preprocessed data to ' + save_path)


if __name__ == '__main__':
    root_dir = '/datasets_master/virtual_kitti'
    out_dir = '/datasets_local/datasets_mjaritz/virtual_kitti_preprocess'
    preprocess('train', root_dir, out_dir)
