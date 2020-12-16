
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
from tqdm import tqdm
import my_optim
from ss_datalayer import SSDatalayer
from oneshot import *
from utils.Restore import restore
from utils import AverageMeter
from utils.save_atten import SAVE_ATTEN
from utils.segscorer import SegScorer

from utils import Metrics

ROOT_DIR = '/'.join(os.getcwd().split('/'))
print ROOT_DIR

save_atten = SAVE_ATTEN()

# SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots')
SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots_15shot_back_non_updown')
DISP_INTERVAL = 20
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str,default='onemodel_sg_one')
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default='/home/liruimin/sg-one-master-coco/snapshots_15shot_back_non_updown')
    parser.add_argument("--lr", type=float, default=1e-5)
    # parser.add_argument("--group", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--restore_step', type=int, default=80000)
    parser.add_argument('--isTest', type=bool, default=False)
    return parser.parse_args()

def measure(y_in, pred_in):
    # thresh = .5
    thresh = .5
    y = y_in>thresh
    pred = pred_in>thresh
    tp = np.logical_and(y,pred).sum()
    tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(y), pred).sum()
    fn = np.logical_and(y, np.logical_not(pred)).sum()
    return tp, tn, fp, fn

def restore(args, model, group):
    savedir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(group, args.num_folds))
    filename='step_%d.pth.tar'%(args.restore_step)
    snapshot = os.path.join(savedir, filename)
    assert os.path.exists(snapshot), "Snapshot file %s does not exist."%(snapshot)

    checkpoint = torch.load(snapshot)
    model.load_state_dict(checkpoint['state_dict'])

    print('Loaded weights from %s'%(snapshot))

def get_model(args):
    model = eval(args.arch).OneModel(args)

    model = model.cuda()

    opti_A = my_optim.get_filter_adam(args, model)

    return model, opti_A

def warper_img(img):
    img_tensor = torch.Tensor(img).cuda()
    img_var = Variable(img_tensor)
    img_var = torch.unsqueeze(img_var, dim=0)
    return img_var

def val(args):


    model, optimizer = get_model(args)

    model.eval()

    num_classes = 20
    tp_list = [0]*num_classes
    fp_list = [0]*num_classes
    fn_list = [0]*num_classes
    iou_list = [0]*num_classes

    hist = np.zeros((21, 21))

    scorer = SegScorer(num_classes=21)

    for group in range(4):
        datalayer = SSDatalayer(group, 5)
        restore(args, model, group)

        for count in tqdm(range(1000)):
            dat = datalayer.dequeue()
            query_img = dat['second_img']
            ref_img = dat['first_img']
            ref_label = dat['second_label']
            query_label = dat['first_label']

            deploy_info = dat['deploy_info']
            semantic_label = deploy_info['first_semantic_labels'][0][0] - 1



            query_img, query_label = torch.Tensor(query_img[0]).cuda(), torch.Tensor(query_label[0][0,:,:]).cuda()

            query_img_var = Variable(query_img)
            query_label_var = Variable(query_label)


            query_img_var = torch.unsqueeze(query_img_var, dim=0)
            query_label_var = torch.unsqueeze(query_label_var, dim=0)

            ref_img_var_list = [warper_img(img) for img in ref_img]
            ref_label_var_list = [label for label in ref_label]


            sup_label = [warper_img(img) for img in ref_label_var_list]
            s_mask_z = sup_label[0]
            s_mask_o = sup_label[1]
            s_mask_t = sup_label[2]
            s_mask_s = sup_label[3]
            s_mask_f = sup_label[4]



            logits_A, logits_B = model.forward_5shot_backnol(query_img_var, ref_img_var_list, ref_label_var_list)       # 1*2*63*47

            loss_val_A0, cluster_loss, loss_bce = model.get_loss(logits_A[0], s_mask_z)
            loss_val_A1, cluster_loss, loss_bce = model.get_loss(logits_A[1], s_mask_o)
            loss_val_A2, cluster_loss, loss_bce = model.get_loss(logits_A[2], s_mask_t)
            loss_val_A3, cluster_loss, loss_bce = model.get_loss(logits_A[3], s_mask_s)
            loss_val_A4, cluster_loss, loss_bce = model.get_loss(logits_A[4], s_mask_f)

            loss_val = (loss_val_A0 + loss_val_A1 + loss_val_A2 + loss_val_A3 + loss_val_A4)/5

            # optimizer.zero_grad()
            # loss_val.backward()
            # optimizer.step()

            values, pred = model.get_pred(logits_B, query_img_var)
            pred = pred.data.cpu().numpy()

            query_label = query_label.cpu().numpy()
            class_ind = int(deploy_info['first_semantic_labels'][0][0])-1 # because class indices from 1 in data layer
            scorer.update(pred, query_label, class_ind+1)
            tp, tn, fp, fn = measure(query_label, pred)
            # iou_img = tp/float(max(tn+fp+fn,1))
            tp_list[class_ind] += tp
            fp_list[class_ind] += fp
            fn_list[class_ind] += fn
            # max in case both pred and label are zero
            iou_list = [tp_list[ic] /
                        float(max(tp_list[ic] + fp_list[ic] + fn_list[ic],1))
                        for ic in range(num_classes)]     # ic: for 1 to 19


            tmp_pred = pred
            tmp_pred[tmp_pred>0.5] = class_ind+1
            tmp_gt_label = query_label
            tmp_gt_label[tmp_gt_label>0.5] = class_ind+1

            hist += Metrics.fast_hist(tmp_pred, query_label, 21)


        print("-------------GROUP %d-------------"%(group))
        print iou_list
        class_indexes = range(group*5, (group+1)*5)
        print 'Mean:', np.mean(np.take(iou_list, class_indexes))

    print('BMVC IOU', np.mean(np.take(iou_list, range(0,20))))

    miou = Metrics.get_voc_iou(hist)
    print('IOU:', miou, np.mean(miou))

    scores = scorer.score()
    for k in scores.keys():
        print(k, np.mean(scores[k]), scores[k])

if __name__ == '__main__':
    args = get_arguments()
    print 'Running parameters:\n'
    print json.dumps(vars(args), indent=4, separators=(',', ':'))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    val(args)
