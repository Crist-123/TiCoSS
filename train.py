from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import os
from core.utils.utils import KLDivLoss
import core.stereo_datasets as datasets
import torch.nn.functional as F
import time
from core.JLNet import JLnet
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass
def boundary(label):
    size = 10
    num_classes=19
    label = F.pad(label.float(), pad=(size, size, size, size), mode="replicate")
    one_hot_label = F.one_hot(label.long(), num_classes=num_classes).permute(0, 3, 1, 2)
    avg_pool2d = nn.AvgPool2d(kernel_size=2*size+1, stride=1, padding=size)
    filtered_label = avg_pool2d(one_hot_label.float())
    mapped_label = torch.exp(-(4*filtered_label - 2)**2)
    max_label, _ = torch.max(mapped_label, dim=1)
    max_label = max_label[:, size:-size, size:-size]
    return max_label.unsqueeze(0)


def sequence_loss(flow_preds, flow_gt, valid, mask,loss_gamma=0.9, max_flow=192):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        # i_loss *= mask
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir='expruns')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def threshold_adaption(i_batch):
    return 1.0

def train(args):

    alpha = 0.1
    
    model = nn.DataParallel(JLnet(args))
    # model_r=nn.DataParallel(RAFTStereo(args,False))

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler)
   
    logging.info("Loading checkpoint...")
    checkpoint = torch.load('./vkitti_pretrained.pth')
    new_checkpoint = {}
    for key, value in checkpoint.items():
        if 'seg_decoder' not in key:
            new_key = 'module.raft.' + key[len('module.'):]
            new_checkpoint[new_key] = value
    model.load_state_dict(new_checkpoint, strict=False)
    logging.info(f"Done loading checkpoint")

    model.cuda()
    model.train()
    model.module.freeze_bn() # We keep BatchNorm frozen

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:

        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2,image1_1,image2_1,flow, valid, label = [x.cuda() for x in data_blob]
            assert model.training
            threshold=1.0

            flow_predictions, disp, weight_matrix, *outputs = model(image1, image2,image1_1,image2_1,threshold, iters=args.train_iters)
            weight_matrix=weight_matrix*args.factor
            ce_loss = []
            kl_loss = []
            criterion = nn.CrossEntropyLoss(reduction='none').cuda()
            softmax_outputs = [F.softmax(output / args.temperature, dim=1).view(-1, 19) for output in outputs]
            ce_loss_unfiltered = [criterion(output, label) for output in outputs]

            if total_steps <10000:
                label_bd = boundary(label)
                weight1=(1-args.scgalpha)*weight_matrix 
                weight2=args.scgalpha*label_bd
                ce_loss_weighted = [loss * (weight1+weight2) for loss in ce_loss_unfiltered]
                ce_loss = [loss.mean() for loss in ce_loss_weighted]
                ce_loss=sum(ce_loss)
                # ce_loss_weighted = [loss * weight_matrix for loss in ce_loss_unfiltered]
                # ce_loss = [loss.mean() for loss in ce_loss_weighted]
                # ce_loss=sum(ce_loss)
            else:
                label_bd = boundary(label)
                weight1=(1-args.scgalpha)*weight_matrix 
                weight2=args.scgalpha*label_bd
                ce_loss_weighted = [loss * (weight1+weight2) for loss in ce_loss_unfiltered]
                ce_loss = [loss.mean() for loss in ce_loss_weighted]
                ce_loss=sum(ce_loss)
            if args.mimic:
                for i, output_flat in enumerate(softmax_outputs):
                    for j, output2_flat in enumerate(softmax_outputs):
                        if i != j:
                            kl_loss.append(args.alpha * KLDivLoss(output_flat, output2_flat.detach(), args.temperature).cuda())
            kl_loss=sum(kl_loss)
            loss_seg = ce_loss + kl_loss

            
            loss_disp, metrics = sequence_loss(flow_predictions, flow, valid,mask=weight_matrix)
            
            loss = loss_disp + loss_seg
            
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            total_steps += 1.0
            if total_steps > args.num_steps:
                should_keep_training = False
                break

    print("FINISHED TRAINING")
    logger.close()
    save_path = Path('checkpoints/%s.pth' % args.name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    return save_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='exp', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="restore checkpoint")
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--is_train', default=True, help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['kitti'], help="training datasets.")
    parser.add_argument('--data_root', default='/home/mias/gftang/data/Joint_learning_data/myKITTI_sem', help="root directory of the KITTI-style training dataset.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=20000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 512], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=32, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=1e-5, help="Weight decay in optimizer.")
    parser.add_argument('--mimic', default=True,
                    help='introduce mimicking losses from different branches')
    parser.add_argument('--temperature', default=1, type=float,
                        help='temperature for smoothing the soft target')
    parser.add_argument('--alpha', default=1, type=float,
                        help='weight of KL divergence loss')
    parser.add_argument('--scgalpha', default=0.1, type=float,
                        help='weight of scg loss')
    parser.add_argument('--scale', type=float,default=1, help="name your experiment")
    parser.add_argument('--factor', type=float,default=1.5, help="name your experiment")
    parser.add_argument('--threshold',type=float, default=1.0, help='use mixed precision')
    parser.add_argument('--punishment',type=float, default=0.5, help='use mixed precision')
    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=22, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path("checkpoints").mkdir(exist_ok=True, parents=True)

    start_time = time.time()
    train(args)
    end_time=time.time()
    total_time = (end_time - start_time)/3600
    print(f"Program total running time: {total_time} hour.")
