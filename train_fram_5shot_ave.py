
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
import shutil
import argparse
import my_optim
from oneshot import *
from utils.LoadDataSeg import data_loader
from utils.Restore import restore
from utils import AverageMeter
from utils.para_number import get_model_para_number


from torchvision.transforms import Compose
from utils.transforms.myTransforms import RandomMirror, Resize, ToTensorNormalize
from utils.utils import set_seed, CLASS_LABELS
from torch.utils.data import DataLoader
from utils.customized import voc_fewshot



os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '9'

# SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots')
ROOT_DIR = '/'.join(os.getcwd().split('/'))  #'/home/liruimin/SG-One-master'
print (ROOT_DIR)

SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots_15shot_back_non_updown')

LR = 1e-5

def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str,default='onemodel_sg_one')
    parser.add_argument("--max_steps", type=int, default=100001)
    parser.add_argument("--lr", type=float, default=LR)
    # parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=20000)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    # parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--start_count", type=int, default=0)

    parser.add_argument("--split", type=str, default='nwns_train')
    parser.add_argument("--group", type=int, default=3)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--isTest', type=bool, default=False)

    return parser.parse_args()


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savedir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(args.group, args.num_folds))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savepath = os.path.join(savedir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):

    model = eval(args.arch).OneModel(args)

    model = model.cuda()

    print('Number of Parameters: %d'%(get_model_para_number(model)))

    # optimizer
    opti_A = my_optim.get_finetune_optimizer(args, model)

    # if os.path.exists(args.restore_from):

    snapshot_dir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(args.group, args.num_folds))
    print(args.resume)
    if args.resume:
        restore(snapshot_dir, model)
        print("Resume training...")

    return model, opti_A

def get_save_dir(args):
    snapshot_dir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(args.group, args.num_folds))
    return snapshot_dir



def train(args):

    losses = AverageMeter()
    model, optimizer= get_model(args)
    model.train()

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    train_loader = data_loader(args)



    if not os.path.exists(get_save_dir(args)):
        os.makedirs(get_save_dir(args))

    save_log_dir = get_save_dir(args)
    log_file  = open(os.path.join(save_log_dir, 'log.txt'),'w')

    count = args.start_count
    for dat in train_loader:
        if count > args.max_steps:
            break
# anchor_img=1*3*333*500; anchor_mask = 1*333*500; pos_img=1*3*375*500; pos_mask=1*375*500; neg_img=1*3*375*500; neg_mask=1*375*500;
        anchor_img, anchor_mask, pos_img, pos_mask = dat

        anchor_img, anchor_mask, = anchor_img[0].cuda(), anchor_mask[0].cuda()

        anchor_mask = torch.unsqueeze(anchor_mask,dim=1)  #1*1*333*500
        s_mask_z = pos_mask[0].cuda().unsqueeze(1)
        s_mask_o = pos_mask[1].cuda().unsqueeze(1)
        s_mask_t = pos_mask[2].cuda().unsqueeze(1)
        s_mask_s = pos_mask[3].cuda().unsqueeze(1)
        s_mask_f = pos_mask[4].cuda().unsqueeze(1)

        logits_A, logits_B = model.forward_5shot_backnol(anchor_img, pos_img, pos_mask)

        loss_val_A0, cluster_loss, loss_bce = model.get_loss(logits_A[0], s_mask_z)
        loss_val_A1, cluster_loss, loss_bce = model.get_loss(logits_A[1], s_mask_o)
        loss_val_A2, cluster_loss, loss_bce = model.get_loss(logits_A[2], s_mask_t)
        loss_val_A3, cluster_loss, loss_bce = model.get_loss(logits_A[3], s_mask_s)
        loss_val_A4, cluster_loss, loss_bce = model.get_loss(logits_A[4], s_mask_f)

        loss_val_A = (loss_val_A0 + loss_val_A1 + loss_val_A2 + loss_val_A3 + loss_val_A4)/5
        loss_val_B, cluster_loss, loss_bce = model.get_loss(logits_B, anchor_mask)

        loss_val_float = loss_val_B.data.item()

        losses.update(loss_val_float, 1)

        out_str = '%d, %.4f\n'%(count, loss_val_float)
        log_file.write(out_str)

        loss_val = loss_val_B

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()


        count += 1
        if count%args.disp_interval == 0:
            # print('Step:%d \t Loss:%.3f '%(count, losses.avg))
            print('Step:%d \t Loss:%.3f \t '
                  'Part1: %.3f \t Part2: %.3f'%(count, losses.avg,
                                        cluster_loss.cpu().data.numpy() if isinstance(cluster_loss, torch.Tensor) else cluster_loss,
                                        loss_bce.cpu().data.numpy() if isinstance(loss_bce, torch.Tensor) else loss_bce))


        if count%args.save_interval == 0 and count >0:
            save_checkpoint(args,
                            {
                                'global_counter': count,
                                'state_dict':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            }, is_best=False,
                            filename='step_%d.pth.tar'
                                     %(count))
    log_file.close()

if __name__ == '__main__':
    args = get_arguments()
    print ('Running parameters:\n')
    print (json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    train(args)
