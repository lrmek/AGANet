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

from ss_datalayer_2way import SSDatalayer
from oneshot import *
from utils.Restore import restore
from utils import AverageMeter
from utils.save_atten import SAVE_ATTEN
from utils.segscorer import SegScorer
from utils.LoadDataSeg import val_loader
from utils import Metrics

ROOT_DIR = '/'.join(os.getcwd().split('/'))
print ROOT_DIR

save_atten = SAVE_ATTEN()

# SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots')
SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots_2way_5shot')
DISP_INTERVAL = 20
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str,default='onemodel_sg_one')
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default='/home/liruimin/SG-One-master/snapshots_2way_5shot')

    parser.add_argument("--group", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--restore_step', type=int, default=160000)
    parser.add_argument('--isTest', type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-5)
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

def warper_img(img):
    img_tensor = torch.Tensor(img).cuda()
    img_var = Variable(img_tensor)
    img_var = torch.unsqueeze(img_var, dim=0)
    return img_var


def val(args):
    model, optimizer = get_model(args)
    losses = AverageMeter()
    #model = get_model(args)
    model.eval()

    num_classes = 20
    tp_list = [0]*num_classes
    fp_list = [0]*num_classes
    fn_list = [0]*num_classes
    iou_list = [0]*num_classes

    hist = np.zeros((21, 21))

    scorer = SegScorer(num_classes=21)
    #test_loader = val_loader(args)


    for group in range(4):
        datalayer = SSDatalayer(group, k_shot=5)
        restore(args, model, group)

        for count in tqdm(range(1000)):
            dat = datalayer.dequeue()
            s_class_one = dat['first_img'][0]
            s_one_label = dat['first_label'][0]
            s_class_two = dat['first_img'][1]
            s_two_label = dat['first_label'][1]
            query_img = dat['second_img'][0]
            query_label = dat['second_label'][0]

            s1_img_list = [warper_img(img) for img in s_class_one]
            s1_label_list = [warper_img(img) for img in s_one_label]
            s2_img_list = [warper_img(img) for img in s_class_two]
            s2_label_list = [warper_img(img) for img in s_two_label]
            query_img, query_label = torch.Tensor(query_img).cuda(), torch.Tensor(query_label[0, :, :]).cuda()

            support_img_var = []
            support_img_var.append(s1_img_list)
            support_img_var.append(s2_img_list)
            support_label_var = []
            support_label_var.append(s1_label_list)
            support_label_var.append(s2_label_list)

            query_img = torch.unsqueeze(query_img, dim=0)  # 1*3*375*500



            deploy_info = dat['deploy_info']
            semantic_label = deploy_info['second_semantic_labels'][0] - 1

            logits_A, logits_B = model.forward_2way_5shot_cat(query_img, support_img_var, support_label_var)

            loss_cls1_one, cluster_loss, loss_bce = model.get_2way_loss(logits_A[0][0], support_label_var[0][0])
            loss_cls1_two, cluster_loss, loss_bce = model.get_2way_loss(logits_A[0][1], support_label_var[0][1])
            loss_cls1_three, cluster_loss, loss_bce = model.get_2way_loss(logits_A[0][2], support_label_var[0][2])
            loss_cls1_four, cluster_loss, loss_bce = model.get_2way_loss(logits_A[0][3], support_label_var[0][3])
            loss_cls1_five, cluster_loss, loss_bce = model.get_2way_loss(logits_A[0][4], support_label_var[0][4])

            loss_cls2_one, cluster_loss, loss_bce = model.get_2way_loss(logits_A[1][0], support_label_var[1][0])
            loss_cls2_two, cluster_loss, loss_bce = model.get_2way_loss(logits_A[1][1], support_label_var[1][1])
            loss_cls2_three, cluster_loss, loss_bce = model.get_2way_loss(logits_A[1][2], support_label_var[1][2])
            loss_cls2_four, cluster_loss, loss_bce = model.get_2way_loss(logits_A[1][3], support_label_var[1][3])
            loss_cls2_five, cluster_loss, loss_bce = model.get_2way_loss(logits_A[1][4], support_label_var[1][4])

            loss_val_A = (loss_cls1_one + loss_cls1_two + loss_cls1_three + loss_cls1_four + loss_cls1_five + loss_cls2_one + loss_cls2_two + loss_cls2_three + loss_cls2_four + loss_cls2_five) / 10

            loss_val = loss_val_A

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()





            values, pred = model.get_pred(logits_B, query_img)    #values=375*500;  pred=375*500
            w, h = query_label.size()
            pred = pred.view(w, h)
            pred = pred.data.cpu().numpy().astype(np.int32)           #187500

            query_label = query_label.data.cpu().numpy().astype(np.int32)  #187500
            class_ind = int(deploy_info['second_semantic_labels'][0]) - 1 # because class indices from 1 in data layer
            scorer.update(pred, query_label, class_ind+1)
            tp, tn, fp, fn = measure(query_label, pred)
            # iou_img = tp/float(max(tn+fp+fn,1))
            tp_list[class_ind] += tp
            fp_list[class_ind] += fp
            fn_list[class_ind] += fn
            # max in case both pred and label are zero
            iou_list = [tp_list[ic] /
                        float(max(tp_list[ic] + fp_list[ic] + fn_list[ic],1))
                        for ic in range(num_classes)]


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


    binary_hist = np.array((hist[0, 0], hist[0, 1:].sum(),hist[1:, 0].sum(), hist[1:, 1:].sum())).reshape((2, 2))
    bin_iu = np.diag(binary_hist) / (binary_hist.sum(1) + binary_hist.sum(0) - np.diag(binary_hist))
    print('Bin_iu:', bin_iu)

    # scores = scorer.score()
    # for k in scores.keys():
    #     print(k, np.mean(scores[k]), scores[k])

if __name__ == '__main__':
    args = get_arguments()
    print 'Running parameters:\n'
    print json.dumps(vars(args), indent=4, separators=(',', ':'))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    val(args)
