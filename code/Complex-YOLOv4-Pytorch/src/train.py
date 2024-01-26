import time
import numpy as np
import sys
import random
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm
from torchsummary import summary

sys.path.append('./')

from data_process.kitti_dataloader import create_train_dataloader, create_val_dataloader
from models.model_utils import create_model, make_data_parallel, get_num_parameters
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint
from utils.train_utils import reduce_tensor, to_python_float, get_tensorboard_log
from utils.misc import AverageMeter, ProgressMeter
from utils.logger import Logger
from config.train_config import parse_train_configs
from evaluate import evaluate_mAP
from ultralytics import YOLO
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.engine.model import DetectionModel
from ultralytics.yolo.v8.detect.train import Loss, DetectionTrainer
from ultralytics.yolo.utils.torch_utils import de_parallel
from ultralytics.yolo.v8.detect.train import *
class custom_loss(Loss):
    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        # print(batch.shape)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        # print(batch_size)
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        # print(batch[2]['cls'])
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = torch.cat()
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


def main():
    configs = parse_train_configs()

    # Re-produce results
    if configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if configs.gpu_idx is not None:
        print('You have chosen a specific GPU. This will completely disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        main_worker(configs.gpu_idx, configs)


def main_worker(gpu_idx, configs):
    configs.gpu_idx = gpu_idx
    configs.device = torch.device('cpu' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx))

    if configs.distributed:
        if configs.dist_url == "env://" and configs.rank == -1:
            configs.rank = int(os.environ["RANK"])
        if configs.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)
        configs.subdivisions = int(64 / configs.batch_size / configs.ngpus_per_node)
    else:
        configs.subdivisions = int(64 / configs.batch_size)

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    if configs.is_master_node:
        logger = Logger(configs.logs_dir, configs.saved_fn)
        logger.info('>>> Created a new logger')
        logger.info('>>> configs: {}'.format(configs))
        tb_writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))
    else:
        logger = None
        tb_writer = None

    # model
    # model = create_model(configs)
    trainer = DetectionTrainer('model-defaults.yaml')
    trainer.setup_model()
    trainer.set_model_attributes()
    model = trainer.model
    # model = trainer.get_model('model-defaults.yaml')
    # model = DetectionTrainer('model-defaults.yaml')
    # model.setup_model()
    # model.set_model_attributes()
    # model = trainer.model()
    # model = trainer.model()
    # loss_fn = Loss(model)
    # print(model.model.modules())
    # load weight from a checkpoint
    if configs.pretrained_path is not None:
        assert os.path.isfile(configs.pretrained_path), "=> no checkpoint found at '{}'".format(configs.pretrained_path)
        model.load_state_dict(torch.load(configs.pretrained_path))
        if logger is not None:
            logger.info('loaded pretrained model at {}'.format(configs.pretrained_path))

    # resume weights of model from a checkpoint
    if configs.resume_path is not None:
        assert os.path.isfile(configs.resume_path), "=> no checkpoint found at '{}'".format(configs.resume_path)
        model.load_state_dict(torch.load(configs.resume_path))
        if logger is not None:
            logger.info('resume training model from checkpoint {}'.format(configs.resume_path))

    # Data Parallel
    model = make_data_parallel(model, configs)

    # Make sure to create optimizer after moving the model to cuda
    optimizer = create_optimizer(configs, model.module)
    # x = model.get
    # optimizer  = model.module.build_optimizer(x)
    lr_scheduler = create_lr_scheduler(optimizer, configs)
    configs.step_lr_in_epoch = True if configs.lr_type in ['multi_step'] else False

    # resume optimizer, lr_scheduler from a checkpoint
    if configs.resume_path is not None:
        utils_path = configs.resume_path.replace('Model_', 'Utils_')
        assert os.path.isfile(utils_path), "=> no checkpoint found at '{}'".format(utils_path)
        utils_state_dict = torch.load(utils_path, map_location='cuda:{}'.format(configs.gpu_idx))
        optimizer.load_state_dict(utils_state_dict['optimizer'])
        lr_scheduler.load_state_dict(utils_state_dict['lr_scheduler'])
        configs.start_epoch = utils_state_dict['epoch'] + 1

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader
    train_dataloader, train_sampler = create_train_dataloader(configs)
    # print(train_dataloader.dataset.__getitem__(0))
    if logger is not None:
        logger.info('number of batches in training set: {}'.format(len(train_dataloader)))

    if configs.evaluate:
        val_dataloader = create_val_dataloader(configs)
        precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, None)
        print('Evaluate - precision: {}, recall: {}, AP: {}, f1: {}, ap_class: {}'.format(precision, recall, AP, f1,
                                                                                          ap_class))
        print('mAP {}'.format(AP.mean()))
        return

    for epoch in range(configs.start_epoch, configs.num_epochs + 1):
        if logger is not None:
            logger.info('{}'.format('*-' * 40))
            logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.num_epochs, '=' * 35))
            logger.info('{}'.format('*-' * 40))
            logger.info('>>> Epoch: [{}/{}]'.format(epoch, configs.num_epochs))

        if configs.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer, trainer)
        if not configs.no_val:
            val_dataloader = create_val_dataloader(configs)
            print('number of batches in val_dataloader: {}'.format(len(val_dataloader)))
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, logger)
            val_metrics_dict = {
                'precision': precision.mean(),
                'recall': recall.mean(),
                'AP': AP.mean(),
                'f1': f1.mean(),
                'ap_class': ap_class.mean()
            }
            if tb_writer is not None:
                tb_writer.add_scalars('Validation', val_metrics_dict, epoch)

        # Save checkpoint
        if configs.is_master_node and ((epoch % configs.checkpoint_freq) == 0):
            model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
            save_checkpoint(configs.checkpoints_dir, configs.saved_fn, model_state_dict, utils_state_dict, epoch)

        if not configs.step_lr_in_epoch:
            lr_scheduler.step()
            if tb_writer is not None:
                tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], epoch)

    if tb_writer is not None:
        tb_writer.close()
    if configs.distributed:
        cleanup()


def cleanup():
    dist.destroy_process_group()


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer, trainer):
    # loss_fn = Loss(YOLO().model)
    # print(f"Model args, {model.modules()}")
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

    num_iters_per_epoch = len(train_dataloader)

    # switch to train mode
    model.train()
    # start_time = time.time()
    # for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
    #     # print(f"This is batch idx, {batch_idx}")
    #     # print(f"this is batch)data, {batch_data}")
    #     print(batch_idx)
    #     data_time.update(time.time() - start_time)
    #     # print(batch_data)
    #     _, imgs, targets = batch_data
    #     # print(targets)
    #     global_step = num_iters_per_epoch * (epoch - 1) + batch_idx + 1
    #
    #     batch_size = imgs.size(0)
    #     # targets = torch.cat((torch.tensor(batch_idx), torch.tensor(targets[:, 1]), torch.tensor(targets[:, 2:6])))
    #     targets = targets.to(configs.device, non_blocking=True).float()
    #     imgs = imgs.to(configs.device, non_blocking=True).float()
    #     # print(imgs.dim())
    #     # loss = model.loss
    #     # print(outputs)
    #     # print(targets.shape)
    #     # print(model.module.get_parameter())
    #     # print(f"the model summary, {summary(model, (3, 608, 608))}")
    #     outputs = model(imgs)
    #     print(outputs[2].shape)
    #     # args ={'hi': 0.3}
    #     #### Break ###
    #     # print(f"The target shape is{targets.shape}")
    #     # print(targets)
    #
    #     loss_fn = Loss(de_parallel(model))
    #     # targets = loss_fn.preprocess(targets, batch_data, 1)
    #     # print(targets.data)
    #     total_loss = loss_fn(outputs, targets)
    #     # total_loss = loss_fn(model(imgs), targets)
    #
    #     # print(type(outputs))
    #     # print(len(outputs))
    #     # For torch.nn.DataParallel case
    #     if (not configs.distributed) and (configs.gpu_idx is None):
    #         total_loss = torch.mean(total_loss)
    #
    #     # compute gradient and perform backpropagation
    #     total_loss.backward()
    #     if global_step % configs.subdivisions == 0:
    #         optimizer.step()
    #         # Adjust learning rate
    #         if configs.step_lr_in_epoch:
    #             lr_scheduler.step()
    #             if tb_writer is not None:
    #                 tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], global_step)
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #     if configs.distributed:
    #         reduced_loss = reduce_tensor(total_loss.data, configs.world_size)
    #     else:
    #         reduced_loss = total_loss.data
    #     losses.update(to_python_float(reduced_loss), batch_size)
    #     # measure elapsed time
    #     # torch.cuda.synchronize()
    #     batch_time.update(time.time() - start_time)
    #
    #     if tb_writer is not None:
    #         if (global_step % configs.tensorboard_freq) == 0:
    #             tensorboard_log = get_tensorboard_log(model)
    #             tb_writer.add_scalar('avg_loss', losses.avg, global_step)
    #             for layer_name, layer_dict in tensorboard_log.items():
    #                 tb_writer.add_scalars(layer_name, layer_dict, global_step)
    #
    #     # Log message
    #     if logger is not None:
    #         if (global_step % configs.print_freq) == 0:
    #             logger.info(progress.get_message(batch_idx))
    #
    #     start_time = time.time()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            cleanup()
            sys.exit(0)
        except SystemExit:
            os._exit(0)
