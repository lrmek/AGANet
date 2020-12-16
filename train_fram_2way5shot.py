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

SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots_2way_5shot')

LR = 1e-5

def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str,default='onemodel_sg_one')
    parser.add_argument("--max_steps", type=int, default=200001)
    parser.add_argument("--lr", type=float, default=LR)
    # parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=20000)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    # parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--start_count", type=int, default=0)

    parser.add_argument("--split", type=str, default='w2t5_train')
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

def warper_img(img):
    img_tensor = torch.Tensor(img).cuda()
    img_var = Variable(img_tensor)
    #img_var = torch.unsqueeze(img_var, dim=0)
    return img_var
def warper_label(lab):
    lab_tensor = torch.Tensor(lab).cuda()
    lab_var = Variable(lab_tensor)
    lab_var = torch.unsqueeze(lab_var, dim=0)
    return lab_var

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

        anchor_img, anchor_mask, pos_img, pos_mask = dat
        s_class_one = pos_img[0]
        s_class_two = pos_img[1]
        s_one_label = pos_mask[0]
        s_two_label = pos_mask[1]

        s1_img_list = [warper_img(img) for img in s_class_one]
        s1_label_list = [warper_label(img) for img in s_one_label]
        s2_img_list = [warper_img(img) for img in s_class_two]
        s2_label_list = [warper_label(img) for img in s_two_label]

        query_img = [warper_img(img) for img in anchor_img][0]
        query_mask = [warper_label(img) for img in anchor_mask][0]
        support_img =[]
        support_mask=[]
        support_img.append(s1_img_list)
        support_img.append(s2_img_list)
        support_mask.append(s1_label_list)
        support_mask.append(s2_label_list)

        logits_A, logits_B = model.forward_2way_5shot_cat(query_img, support_img, support_mask)

        loss_cls1_one, cluster_loss, loss_bce = model.get_2way_loss(logits_A[0][0], support_mask[0][0])
        loss_cls1_two, cluster_loss, loss_bce = model.get_2way_loss(logits_A[0][1], support_mask[0][1])
        loss_cls1_three, cluster_loss, loss_bce = model.get_2way_loss(logits_A[0][2], support_mask[0][2])
        loss_cls1_four, cluster_loss, loss_bce = model.get_2way_loss(logits_A[0][3], support_mask[0][3])
        loss_cls1_five, cluster_loss, loss_bce = model.get_2way_loss(logits_A[0][4], support_mask[0][4])

        loss_cls2_one, cluster_loss, loss_bce = model.get_2way_loss(logits_A[1][0], support_mask[1][0])
        loss_cls2_two, cluster_loss, loss_bce = model.get_2way_loss(logits_A[1][1], support_mask[1][1])
        loss_cls2_three, cluster_loss, loss_bce = model.get_2way_loss(logits_A[1][2], support_mask[1][2])
        loss_cls2_four, cluster_loss, loss_bce = model.get_2way_loss(logits_A[1][3], support_mask[1][3])
        loss_cls2_five, cluster_loss, loss_bce = model.get_2way_loss(logits_A[1][4], support_mask[1][4])

        loss_val_A = (loss_cls1_one+loss_cls1_two+loss_cls1_three+loss_cls1_four+loss_cls1_five+loss_cls2_one+loss_cls2_two+loss_cls2_three+loss_cls2_four+loss_cls2_five)/10


        loss_val_B, cluster_loss, loss_bce = model.get_2way_loss(logits_B, query_mask)

        loss_val_float = loss_val_B.data.item() + loss_val_A.data.item()

        losses.update(loss_val_float, 1)

        out_str = '%d, %.4f\n'%(count, loss_val_float)
        log_file.write(out_str)

        loss_val = loss_val_B + loss_val_A

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
