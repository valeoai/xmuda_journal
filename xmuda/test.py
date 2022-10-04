#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings
import numpy as np

import torch
import torch.nn.functional as F

from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.build import build_model_2d, build_model_3d
from xmuda.data.build import build_dataloader
from xmuda.data.utils.validate import validate


def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('ckpt2d', type=str, help='path to checkpoint file of the 2D model')
    parser.add_argument('ckpt3d', type=str, help='path to checkpoint file of the 3D model')
    parser.add_argument('--pselab', action='store_true', help='generate pseudo-labels')
    parser.add_argument('--save-ensemble', action='store_true',
                        help='Whether to save the 2D+3D ensembling pseudo labels.')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def predict(cfg,
            model_2d,
            model_3d,
            dataloader,
            pselab_path,
            save_ensemble=False):
    """
    Function to save pseudo labels. The difference with the "validate" function is that no ground truth labels are
    required, i.e. no evaluation (mIoU) is computed.

    :param cfg: Configuration node.
    :param model_2d: 2D model.
    :param model_3d: 3D model. Optional.
    :param dataloader: Dataloader for the test dataset.
    :param pselab_path: Path to dictionary where to save the pseudo labels as npy file.
    :param save_ensemble: Whether to save the ensemble labels (2D+3D).
    """
    logger = logging.getLogger('xmuda.predict')
    logger.info('Prediction of Pseudo Labels')

    if not pselab_path:
        raise ValueError('A pseudo label path must be provided for this function.')
    if not model_2d:
        raise ValueError('A 2D model must be provided.')
    if save_ensemble and not model_3d:
        raise ValueError('For ensembling, a 3D model needs to be provided.')
    pselab_data_list = []

    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch) if model_3d else None

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy()
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if model_3d else None

            # get original point cloud from before voxelization
            points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            for batch_ind in range(len(points_idx)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if model_3d else None

                # pseudo label
                assert np.all(pred_label_2d >= 0)
                curr_probs_2d = probs_2d[left_idx:right_idx]
                curr_probs_3d = probs_3d[left_idx:right_idx] if model_3d else None
                pseudo_label_dict = {
                    'probs_2d': curr_probs_2d[range(len(pred_label_2d)), pred_label_2d].cpu().numpy(),
                    'pseudo_label_2d': pred_label_2d.astype(np.uint8),
                    'probs_3d': curr_probs_3d[range(len(pred_label_3d)), pred_label_3d].cpu().numpy() if model_3d else None,
                    'pseudo_label_3d': pred_label_3d.astype(np.uint8) if model_3d else None
                }
                if save_ensemble:
                    pseudo_label_dict['pseudo_label_ensemble'] = pred_label_ensemble.astype(np.uint8)
                pselab_data_list.append(pseudo_label_dict)

                left_idx = right_idx

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    '  '.join(
                        [
                            'iter: {iter}/{total_iter}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        np.save(pselab_path, pselab_data_list)
        logger.info('Saved pseudo label data to {}'.format(pselab_path))


def test(cfg, args, output_dir=''):
    logger = logging.getLogger('xmuda.test')

    # build 2d model
    model_2d = build_model_2d(cfg)[0]

    # build 3d model
    model_3d = build_model_3d(cfg)[0]

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build checkpointer
    checkpointer_2d = CheckpointerV2(model_2d, save_dir=output_dir, logger=logger)
    if args.ckpt2d:
        # load weight if specified
        weight_path = args.ckpt2d.replace('@', output_dir)
        checkpointer_2d.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer_2d.load(None, resume=True)
    checkpointer_3d = CheckpointerV2(model_3d, save_dir=output_dir, logger=logger)
    if args.ckpt3d:
        # load weight if specified
        weight_path = args.ckpt3d.replace('@', output_dir)
        checkpointer_3d.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer_3d.load(None, resume=True)

    # build dataset
    test_dataloader = build_dataloader(cfg, mode='test', domain='target')

    pselab_path = None
    if args.pselab:
        pselab_dir = osp.join(output_dir, 'pselab_data')
        os.makedirs(pselab_dir, exist_ok=True)
        assert len(cfg.DATASET_TARGET.TEST) == 1
        pselab_path = osp.join(pselab_dir, cfg.DATASET_TARGET.TEST[0] + '.npy')

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #

    set_random_seed(cfg.RNG_SEED)
    test_metric_logger = MetricLogger(delimiter='  ')
    model_2d.eval()
    model_3d.eval()

    if args.pselab:
        predict(cfg, model_2d, model_3d, test_dataloader, pselab_path, args.save_ensemble)
    else:
        validate(cfg, model_2d, model_3d, test_dataloader, test_metric_logger, pselab_path=pselab_path)


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('xmuda', output_dir, comment='test.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    test(cfg, args, output_dir)


if __name__ == '__main__':
    main()
