import torch
import torch.nn as nn

from xmuda.models.resnet34_unet import UNetResNet34
from xmuda.models.scn_unet import UNetSCN


class Net2D3DFusionSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 backbone_2d_kwargs,
                 backbone_3d,
                 backbone_3d_kwargs
                 ):
        super(Net2D3DFusionSeg, self).__init__()

        # 2D image network
        if backbone_2d == 'UNetResNet34':
            self.net_2d = UNetResNet34(**backbone_2d_kwargs)
            feat_channels_2d = 64
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        # 3D network
        if backbone_3d == 'SCN':
            self.net_3d = UNetSCN(**backbone_3d_kwargs)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        # fusion
        self.fuse = nn.Sequential(
            nn.Linear(feat_channels_2d + self.net_3d.out_channels, 64),
            nn.ReLU(inplace=True)
        )
        # segmentation head
        self.linear = nn.Linear(64, num_classes)

        # mimicking heads
        self.dual_head = dual_head
        if dual_head:
            self.linear_2d = nn.Linear(feat_channels_2d, num_classes)
            self.linear_3d = nn.Linear(self.net_3d.out_channels, num_classes)

    def forward(self, data_batch):
        # (batch_size, 3, H, W)
        img = data_batch['img']
        img_indices = data_batch['img_indices']

        # 2D network
        x = self.net_2d(img)

        # 2D-3D feature lifting
        feats_2d = []
        for i in range(x.shape[0]):
            feats_2d.append(x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
        feats_2d = torch.cat(feats_2d, 0)

        # 3D network
        feats_3d = self.net_3d(data_batch['x'])

        # fusion + segmentation
        x = torch.cat([feats_2d, feats_3d], 1)
        feats_fuse = self.fuse(x)
        x = self.linear(feats_fuse)

        preds = {
            'feats': feats_fuse,
            'seg_logit': x,
        }

        # mimicking heads
        if self.dual_head:
            preds['seg_logit_2d'] = self.linear_2d(feats_2d)
            preds['seg_logit_3d'] = self.linear_3d(feats_3d)

        return preds


def test_Net2D3DFusionSeg():
    # 2D
    batch_size = 2
    img_width = 400
    img_height = 225

    # 3D
    num_coords = 2000
    num_classes = 11
    full_scale = 4096
    in_channels = 1

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

    # 3D
    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)

    # to cuda
    feats = feats.cuda()
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_fuse = Net2D3DFusionSeg(num_classes,
                                backbone_2d='UNetResNet34',
                                backbone_2d_kwargs={},
                                backbone_3d='SCN',
                                backbone_3d_kwargs={'in_channels': in_channels},
                                dual_head=True)

    net_fuse.cuda()
    out_dict = net_fuse({
        'img': img,
        'img_indices': img_indices,
        'x': [coords, feats]
    })
    for k, v in out_dict.items():
        print('Net2D3DFusionSeg:', k, v.shape)


if __name__ == '__main__':
    test_Net2D3DFusionSeg()
