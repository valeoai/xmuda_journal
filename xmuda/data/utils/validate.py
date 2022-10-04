import numpy as np
import logging
import time

import torch
import torch.nn.functional as F

from xmuda.data.utils.evaluate import Evaluator


def validate(cfg,
             model_2d,
             model_3d,
             dataloader,
             val_metric_logger,
             pselab_path=None):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Validation')

    dual_model = (model_2d is not None) and (model_3d is not None)

    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names) if model_2d else None
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if dual_model else None

    pselab_data_list = []

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            preds_2d = model_2d(data_batch) if model_2d else None
            preds_3d = model_3d(data_batch) if model_3d else None

            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy() if model_2d else None
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy() if model_3d else None

            # softmax average (ensembling)
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1) if model_2d else None
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy() if dual_model else None

            # get original point cloud from before voxelization
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx] if model_2d else None
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx] if dual_model else None

                # evaluate
                if model_2d:
                    evaluator_2d.update(pred_label_2d, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d, curr_seg_label)
                if dual_model:
                    evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)

                if pselab_path is not None:
                    if model_2d:
                        assert np.all(pred_label_2d >= 0)
                    if model_3d:
                        assert np.all(pred_label_3d >= 0)
                    curr_probs_2d = probs_2d[left_idx:right_idx] if model_2d else None
                    curr_probs_3d = probs_3d[left_idx:right_idx] if model_3d else None
                    pselab_data_list.append({
                        'probs_2d': curr_probs_2d[range(len(pred_label_2d)), pred_label_2d].cpu().numpy() if model_2d else None,
                        'pseudo_label_2d': pred_label_2d.astype(np.uint8)  if model_2d else None,
                        'probs_3d': curr_probs_3d[range(len(pred_label_3d)), pred_label_3d].cpu().numpy() if model_3d else None,
                        'pseudo_label_3d': pred_label_3d.astype(np.uint8) if model_3d else None
                    })

                left_idx = right_idx

            seg_loss_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch['seg_label']) if model_2d else None
            seg_loss_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch['seg_label']) if model_3d else None
            if seg_loss_2d is not None:
                val_metric_logger.update(seg_loss_2d=seg_loss_2d)
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        eval_list = []
        if evaluator_2d is not None:
            val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
            eval_list.append(('2D', evaluator_2d))
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
            eval_list.append(('3D', evaluator_3d))
        if dual_model:
            eval_list.append(('2D+3D', evaluator_ensemble))
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy: {:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU: {:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

        if pselab_path is not None:
            np.save(pselab_path, pselab_data_list)
            logger.info('Saved pseudo label data to {}'.format(pselab_path))


def validate_three_2d_models(
        cfg,
        model_2d1,
        model_2d2,
        model_2d3,
        dataloader,
        val_metric_logger,
        pselab_path=None
):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Validation')

    if model_2d1 is None or model_2d2 is None or model_2d3 is None:
        raise ValueError('All three models must be valid.')

    # evaluator
    class_names = dataloader.dataset.class_names
    evaluator_2d1 = Evaluator(class_names)
    evaluator_2d2 = Evaluator(class_names)
    evaluator_2d3 = Evaluator(class_names)
    evaluator_ensemble = Evaluator(class_names)

    pselab_data_list = []

    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            # copy data from cpu to gpu
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError

            # predict
            preds_2d1 = model_2d1(data_batch)
            preds_2d2 = model_2d2(data_batch)
            preds_2d3 = model_2d3(data_batch)

            pred_label_voxel_2d1 = preds_2d1['seg_logit'].argmax(1).cpu().numpy()
            pred_label_voxel_2d2 = preds_2d2['seg_logit'].argmax(1).cpu().numpy()
            pred_label_voxel_2d3 = preds_2d3['seg_logit'].argmax(1).cpu().numpy()

            # softmax average (ensembling)
            probs_2d1 = F.softmax(preds_2d1['seg_logit'], dim=1)
            probs_2d2 = F.softmax(preds_2d2['seg_logit'], dim=1)
            probs_2d3 = F.softmax(preds_2d3['seg_logit'], dim=1)
            pred_label_voxel_ensemble = (probs_2d1 + probs_2d2 + probs_2d3).argmax(1).cpu().numpy()

            # get original point cloud from before voxelization
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            # loop over batch
            left_idx = 0
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                # check if all points have predictions (= all voxels inside receptive field)
                assert np.all(curr_points_idx)

                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d1 = pred_label_voxel_2d1[left_idx:right_idx]
                pred_label_2d2 = pred_label_voxel_2d2[left_idx:right_idx]
                pred_label_2d3 = pred_label_voxel_2d3[left_idx:right_idx]
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx]

                # evaluate
                evaluator_2d1.update(pred_label_2d1, curr_seg_label)
                evaluator_2d2.update(pred_label_2d2, curr_seg_label)
                evaluator_2d3.update(pred_label_2d3, curr_seg_label)
                evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)

                if pselab_path is not None:
                    assert np.all(pred_label_2d1 >= 0)
                    assert np.all(pred_label_2d2 >= 0)
                    assert np.all(pred_label_2d3 >= 0)
                    curr_probs_2d1 = probs_2d1[left_idx:right_idx]
                    curr_probs_2d2 = probs_2d2[left_idx:right_idx]
                    curr_probs_2d3 = probs_2d3[left_idx:right_idx]
                    current_probs_ensemble = (curr_probs_2d1 + curr_probs_2d2 + curr_probs_2d3) / 3
                    pselab_data_list.append({
                        'probs_2d': current_probs_ensemble[range(len(pred_label_ensemble)), pred_label_ensemble].cpu().numpy(),
                        'pseudo_label_2d': pred_label_ensemble.astype(np.uint8),
                        'probs_3d': None,
                        'pseudo_label_3d': None
                    })

                left_idx = right_idx

            seg_loss_2d1 = F.cross_entropy(preds_2d1['seg_logit'], data_batch['seg_label'])
            seg_loss_2d2 = F.cross_entropy(preds_2d2['seg_logit'], data_batch['seg_label'])
            seg_loss_2d3 = F.cross_entropy(preds_2d3['seg_logit'], data_batch['seg_label'])
            val_metric_logger.update(seg_loss_2d1=seg_loss_2d1)
            val_metric_logger.update(seg_loss_2d2=seg_loss_2d2)
            val_metric_logger.update(seg_loss_2d3=seg_loss_2d3)

            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()

            # log
            cur_iter = iteration + 1
            if cur_iter == 1 or (cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.VAL.LOG_PERIOD == 0):
                logger.info(
                    val_metric_logger.delimiter.join(
                        [
                            'iter: {iter}/{total_iter}',
                            '{meters}',
                            'max mem: {memory:.0f}',
                        ]
                    ).format(
                        iter=cur_iter,
                        total_iter=len(dataloader),
                        meters=str(val_metric_logger),
                        memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                    )
                )

        eval_list = []
        val_metric_logger.update(seg_iou_2d1=evaluator_2d1.overall_iou)
        eval_list.append(('2D_1', evaluator_2d1))
        val_metric_logger.update(seg_iou_2d2=evaluator_2d2.overall_iou)
        eval_list.append(('2D_2', evaluator_2d2))
        val_metric_logger.update(seg_iou_2d3=evaluator_2d3.overall_iou)
        eval_list.append(('2D_3', evaluator_2d3))
        eval_list.append(('2D+3D', evaluator_ensemble))
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy: {:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU: {:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

        if pselab_path is not None:
            np.save(pselab_path, pselab_data_list)
            logger.info('Saved pseudo label data to {}'.format(pselab_path))
