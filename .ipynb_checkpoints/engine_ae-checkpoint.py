import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils
from losses import ReductionAELoss,ReductionAELossWoClassify
import numpy as np
# torch.backends.cudnn.benchmark = True
def train_one_epoch_ae(model: torch.nn.Module, criterion: ReductionAELoss ,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True, surgery=None, output_dir=None, use_classify_loss = False):
    model.train(set_training_mode)
    
            

    if surgery:
        model.module.patch_embed.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.9f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 25
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        samples, targets = batch[0], batch[1]        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            if use_classify_loss:
                logit, f_ori, f_recons = model(samples,targets)
            else:
                f_ori, f_recons = model(samples,targets)
            if use_classify_loss:
                loss, global_loss, ae_loss = criterion(logit,targets, f_ori, f_recons)
            else:
                loss = criterion(f_ori, f_recons)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()  
        
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        
        
        
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        if use_classify_loss:
            metric_logger.update(global_loss=global_loss.item())
            metric_logger.update(ae_loss=ae_loss.item())
        
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_ae(data_loader, model, device, use_classify_loss = False ):
    criterion = ReductionAELossWoClassify()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 50, header):
        images, target = batch[0], batch[1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if use_classify_loss:
                logit, f_ori, f_recons = model(images,targets)
            else:
                f_ori, f_recons = model(images,target)
            
        if use_classify_loss :
            loss,global_loss,ae_loss = criterion(logit,target, f_ori, f_recons)
        else:
            loss = criterion(f_ori, f_recons)
        
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        if use_classify_loss:
            metric_logger.update(global_loss=global_loss.item())
            acc1, acc5 = accuracy(logit, target, topk=(1, 5))
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            metric_logger.meters['ae_loss'].update(ae_loss.item(), n=batch_size)

            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if use_classify_loss:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* loss {losses.global_avg:.3f}'
              .format(losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

