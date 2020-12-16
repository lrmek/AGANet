
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.vgg import vgg_sg as vgg
from oneshot.non_local import NONLocalBlock2D
from oneshot.aspp_block import  _ASPP
from models.resnet import resnet as resnet

class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()
        #self.netB = resnet.resnet101(pretrained=True)
        self.netB = vgg.vgg16(pretrained=True, use_decoder=True)
        if args.isTest:
            for p in self.parameters():
                p.requires_grad = False

        self.classifier_6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, dilation=1,  padding=1),   #fc6
            nn.ReLU(inplace=True),
        )

        self.nol = NONLocalBlock2D(in_channels=256)

        self.aspp = _ASPP(256, 256, [1, 6, 12, 18])


        self.cls = nn.Conv2d(128, 2, kernel_size=1, padding=0)
        self.cls_2way1shot = nn.Conv2d(128, 3, kernel_size=1, padding=0)



        #self.fuse = nn.Conv2d(512, 256, kernel_size=1, padding=0)
        self.fuse =nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256, momentum=1, affine=True),
            nn.ReLU(),
        )



        self.bce_logits_func = nn.CrossEntropyLoss()
        self.loss_func = nn.BCELoss()
        self.cos_similarity_func = nn.CosineSimilarity()
        self.triplelet_func = nn.TripletMarginLoss(margin=2.0)




    def forward(self, anchor_img, pos_img, neg_img, pos_mask):      #query_label_var=1*3*375*500, pos_img=1*3*500*375, neg_img=1*1*500*375, pos_mask = 1*1*500*375
        outA = self.netB(pos_img)  # 1*512*42*63
        outB = self.netB(anchor_img)  # outB=1*512*48*63

        a, b, c, d = outA.size()
        pos_maskdown = F.interpolate(pos_mask, size=(c, d))  # 1*1*42*63
        outA_fore = outA * pos_maskdown  # #1*512*42*63

        vec_low = torch.sum(torch.sum(outA_fore, dim=3), dim=2) / (torch.sum(pos_maskdown) + 1e-5)

        outB_nol = self.nol(outB, outA_fore)  # 1*512*48*63

        _, _, mask_w, mask_h = pos_mask.size()  # pos_mask = 1*1*333*500
        outA_pos = F.upsample(outA, size=(mask_w, mask_h), mode='bilinear')  ##1*512*333*500
        vec_pos = torch.sum(torch.sum(outA_pos * pos_mask, dim=3), dim=2) / torch.sum(pos_mask)  # 1*512

        vec_pos = 0.5 * vec_pos + 0.5 * vec_low
        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)  # 1*512*1*1



        tmp_seg = self.cos_similarity_func(outB_nol, vec_pos)  # tmp_seg = self.cos_similarity_func(outB_nol, vec_pos)             # 1*47*63
        outB_nol = outB_nol * tmp_seg.unsqueeze(dim=1)  # outB_nol = outB_nol * tmp_seg.unsqueeze(dim=1)                #1*256*42*63
        outB_nol = self.aspp(outB_nol)  # 1*256*42*63
        outB_fuse = self.classifier_6(outB_nol)  # 1*256*48*63
        outB_fuse = self.cls(outB_fuse)  # 1*2*48*63

        return  outB_fuse

    def forward_1way_1shot_heat(self, anchor_img, pos_img, neg_img, pos_mask):      #query_label_var=1*3*375*500, pos_img=1*3*500*375, neg_img=1*1*500*375, pos_mask = 1*1*500*375


        outA = self.netB(pos_img)  # 1*512*42*63
        outB = self.netB(anchor_img)  # outB=1*512*48*63
        a, b, c, d = outA.size()
        pos_maskdown = F.interpolate(pos_mask, size=(c, d))  # 1*1*42*63
        outA_fore = outA * pos_maskdown  # #1*512*42*63
        vec_low = torch.sum(torch.sum(outA_fore, dim=3), dim=2) / (torch.sum(pos_maskdown) + 1e-5)
        outB_nol = self.nol(outB, outA_fore)  # 1*512*48*63
        _, _, mask_w, mask_h = pos_mask.size()  # pos_mask = 1*1*333*500
        outA_pos = F.upsample(outA, size=(mask_w, mask_h), mode='bilinear')  ##1*512*333*500
        vec_pos = torch.sum(torch.sum(outA_pos * pos_mask, dim=3), dim=2) / torch.sum(pos_mask)  # 1*512
        vec_pos = 0.5 * vec_pos + 0.5 * vec_low
        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)  # 1*512*1*1

        tmp_seg_A = self.cos_similarity_func(outA,vec_pos)  # tmp_seg = self.cos_similarity_func(outB_nol, vec_pos)             # 1*47*63
        outA_tem = outA * tmp_seg_A.unsqueeze(dim=1)  # outB_nol = outB_nol * tmp_seg.unsqueeze(dim=1)                #1*256*42*63
        outA_spp = self.aspp(outA_tem)  # 1*256*42*63
        outA_fuse = self.classifier_6(outA_spp)  # 1*256*48*63
        outA_fuse = self.cls(outA_fuse)  # 1*2*48*63


        tmp_seg = self.cos_similarity_func(outB_nol, vec_pos)  # tmp_seg = self.cos_similarity_func(outB_nol, vec_pos)             # 1*47*63
        outB_nol = outB_nol * tmp_seg.unsqueeze(dim=1)  # outB_nol = outB_nol * tmp_seg.unsqueeze(dim=1)                #1*256*42*63
        outB_nol = self.aspp(outB_nol)  # 1*256*42*63
        outB_fuse = self.classifier_6(outB_nol)  # 1*256*48*63
        outB_fuse = self.cls(outB_fuse)  # 1*2*48*63

        return  outA_fuse, outB_fuse

    def get_atten_map(self, feature_maps, gt_labels, normalize=True):
        label = gt_labels.long()

        feature_map_size = feature_maps.size()
        batch_size = feature_map_size[0]

        atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]])
        atten_map = Variable(atten_map.cuda())
        for batch_idx in range(batch_size):
            atten_map[batch_idx, :, :] = torch.squeeze(feature_maps[batch_idx, label.data[batch_idx], :, :])

        if normalize:
            atten_map = self.normalize_atten_maps(atten_map)

        return atten_map

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        # --------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,)) - batch_mins, batch_maxs - batch_mins +  1e-5)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed


    def forward_2way_1shot_avg(self, anchor_img, pos_img_list, pos_mask_list):
        outB = self.netB(anchor_img)  # 1*256*63*50

        Noloc_B_sum = []
        tmp_seg_sum = []
        outA_sum = []
        pred_A_sum = []
        for i in range(len(pos_img_list)):
            pos_img = pos_img_list[i]
            pos_mask = pos_mask_list[i]

            for j in range(len(pos_img)):
                pos_img_j = pos_img[j]
                pos_mask_j = pos_mask[j]
                outA_support = self.netB(pos_img_j)  # 1*256*42*63
                outA_sum.append(outA_support)
                a, b, c, d = outA_support.size()  # 1*256*42*63
                pos_maskdown = F.interpolate(pos_mask_j, size=(c, d))  # 1*1*42*63
                outA_down = outA_support * pos_maskdown  # #1*256*4263
                vec_low = torch.sum(torch.sum(outA_down, dim=3), dim=2) / (torch.sum(pos_maskdown) + 1e-5)  # 1*256
                _, _, mask_w, mask_h = pos_mask_j.size()
                outA_pos = F.upsample(outA_support, size=(mask_w, mask_h), mode='bilinear')  # 1*256*375*500
                vec_hig = torch.sum(torch.sum(outA_pos * pos_mask_j, dim=3), dim=2) / torch.sum(pos_mask_j)  # 1*256
                vec_pos = 0.5 * vec_low + 0.5 * vec_hig  # 1*256
                w_b, w_c, w_k, w_g = outA_down.size()
                outA_down = (outA_down).view(w_b, w_c, w_g, w_k)  # 1*256*63*42
                outB_nol = self.nol(outB, outA_down)  # # 1*256*63*50
                Noloc_B_sum.append(outB_nol)
                tmp_seg_sum.append(vec_pos)
        tmp_seg_fuse = (tmp_seg_sum[0] + tmp_seg_sum[1]) / 2
        tmp_seg_fuse = tmp_seg_fuse.unsqueeze(dim=2).unsqueeze(dim=3)

        one_A_seg = self.cos_similarity_func(outA_sum[0], tmp_seg_fuse)
        outA_one_fuse = outA_sum[0] * one_A_seg.unsqueeze(dim=1)
        outA_one_aspp = self.aspp(outA_one_fuse)  # 1*256*63*47
        outA_one_cla = self.classifier_6(outA_one_aspp)  # 1*128*63*47
        pred_one_A = self.cls_2way1shot(outA_one_cla)

        two_A_seg = self.cos_similarity_func(outA_sum[1], tmp_seg_fuse)
        outA_two_fuse = outA_sum[1] * two_A_seg.unsqueeze(dim=1)
        outA_two_aspp = self.aspp(outA_two_fuse)  # 1*256*63*47
        outA_two_cla = self.classifier_6(outA_two_aspp)  # 1*128*63*47
        pred_two_A = self.cls_2way1shot(outA_two_cla)

        outB_fuse = (Noloc_B_sum[0] + Noloc_B_sum[1]) / 2
        tmp_seg = self.cos_similarity_func(outB_fuse, tmp_seg_fuse)
        outB_fuse = outB_fuse * tmp_seg.unsqueeze(dim=1)
        outB_aspp = self.aspp(outB_fuse)  # 1*256*63*47
        outB_cla = self.classifier_6(outB_aspp)  # 1*128*63*47
        pred = self.cls_2way1shot(outB_cla)
        return pred_one_A, pred_two_A, pred

    def forward_2way_1shot_cat(self, anchor_img, pos_img_list, pos_mask_list):
        outB = self.netB(anchor_img)  # 1*256*63*50

        Noloc_B_sum = []
        tmp_seg_sum = []
        outA_sum = []
        pred_A_sum = []
        for i in range(len(pos_img_list)):
            pos_img = pos_img_list[i]
            pos_mask = pos_mask_list[i]

            for j in range(len(pos_img)):
                pos_img_j = pos_img[j]
                pos_mask_j = pos_mask[j]
                outA_support = self.netB(pos_img_j)  # 1*256*42*63
                outA_sum.append(outA_support)
                a, b, c, d = outA_support.size()  # 1*256*42*63
                pos_maskdown = F.interpolate(pos_mask_j, size=(c, d))  # 1*1*42*63
                outA_down = outA_support * pos_maskdown  # #1*256*4263
                vec_low = torch.sum(torch.sum(outA_down, dim=3), dim=2) / (torch.sum(pos_maskdown) + 1e-5)  # 1*256
                _, _, mask_w, mask_h = pos_mask_j.size()
                outA_pos = F.upsample(outA_support, size=(mask_w, mask_h), mode='bilinear')  # 1*256*375*500
                vec_hig = torch.sum(torch.sum(outA_pos * pos_mask_j, dim=3), dim=2) / torch.sum(pos_mask_j)  # 1*256
                vec_pos = 0.5 * vec_low + 0.5 * vec_hig  # 1*256
                w_b, w_c, w_k, w_g = outA_down.size()
                outA_down = (outA_down).view(w_b, w_c, w_g, w_k)  # 1*256*63*42
                outB_nol = self.nol(outB, outA_down)  # # 1*256*63*50
                Noloc_B_sum.append(outB_nol)
                tmp_seg_sum.append(vec_pos)
        tmp_seg_fuse = (tmp_seg_sum[0] + tmp_seg_sum[1]) / 2
        tmp_seg_fuse = tmp_seg_fuse.unsqueeze(dim=2).unsqueeze(dim=3)

        one_A_seg = self.cos_similarity_func(outA_sum[0], tmp_seg_sum[0].unsqueeze(dim=2).unsqueeze(dim=3))
        outA_one_fuse = outA_sum[0] * one_A_seg.unsqueeze(dim=1)
        outA_one_aspp = self.aspp(outA_one_fuse)  # 1*256*63*47
        outA_one_cla = self.classifier_6(outA_one_aspp)  # 1*128*63*47
        pred_one_A = self.cls_2way1shot(outA_one_cla)

        two_A_seg = self.cos_similarity_func(outA_sum[1], tmp_seg_sum[1].unsqueeze(dim=2).unsqueeze(dim=3))
        outA_two_fuse = outA_sum[1] * two_A_seg.unsqueeze(dim=1)
        outA_two_aspp = self.aspp(outA_two_fuse)  # 1*256*63*47
        outA_two_cla = self.classifier_6(outA_two_aspp)  # 1*128*63*47
        pred_two_A = self.cls_2way1shot(outA_two_cla)


        outB_fuse = (Noloc_B_sum[0] + Noloc_B_sum[1])/2
        tmp_seg = self.cos_similarity_func(outB_fuse, tmp_seg_fuse)
        outB_fuse = outB_fuse * tmp_seg.unsqueeze(dim=1)
        outB_aspp = self.aspp(outB_fuse)  # 1*256*63*47
        outB_cla = self.classifier_6(outB_aspp)  # 1*128*63*47
        pred = self.cls_2way1shot(outB_cla)
        return pred_one_A, pred_two_A, pred


    def forward_2way_5shot_cat(self, anchor_img, pos_img_list, pos_mask_list):
        outB = self.netB(anchor_img)  # 1*256*39*63

        vec_all_sum = []
        outB_local_sum = []
        outA_all_sum = []

        for i in range(len(pos_img_list)):
            pos_img = pos_img_list[i]
            pos_mask = pos_mask_list[i]
            pred_outB_cat = []

            outB_one_sum = []
            outA_one_sum = []
            vec_one_sum = []
            for j in range(len(pos_img)):
                pos_img_j = pos_img[j]
                pos_mask_j = pos_mask[j]
                outA_support = self.netB(pos_img_j)  # 1*256*47*63
                outA_one_sum.append(outA_support)

                a, b, c, d = outA_support.size()  # 1*256*48*63
                pos_maskdown = F.interpolate(pos_mask_j, size=(c, d))  # 1*1*47*63
                outA_down = outA_support * pos_maskdown  # #1*256*47*63
                vec_low = torch.sum(torch.sum(outA_down, dim=3), dim=2) / (torch.sum(pos_maskdown) + 1e-5)  # 1*256

                _, _, mask_w, mask_h = pos_mask_j.size()
                outA_pos = F.upsample(outA_support, size=(mask_w, mask_h), mode='bilinear')  # 1*256*375*500
                vec_hig = torch.sum(torch.sum(outA_pos * pos_mask_j, dim=3), dim=2) / torch.sum(pos_mask_j)  # 1*256
                vec_pos = 0.5 * vec_low + 0.5 * vec_hig  # 1*256
                vec_one_sum.append(vec_pos)


                w_b, w_c, w_k, w_g = outA_down.size()
                outA_down = (outA_down).view(w_b, w_c, w_g, w_k)         #1*256*63*47
                outB_nol = self.nol(outB, outA_down)  # 1*256*63*47
                outB_one_sum.append(outB_nol)


                vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)      #1*256*1*1
                tmp_seg = self.cos_similarity_func(outB_nol, vec_pos)    #1*63*47
                outB_nol = outB_nol * tmp_seg.unsqueeze(dim=1)           #1*256*63*47
                pred_outB_cat.append(outB_nol)

            outA_all_sum.append(outA_one_sum)
            qry_nol_one = (outB_one_sum[0] + outB_one_sum[1] +outB_one_sum[2] +outB_one_sum[3] + outB_one_sum[4])/5
            outB_local_sum.append(qry_nol_one)
            vec_class_one = (vec_one_sum[0] + vec_one_sum[1]+ vec_one_sum[2]+ vec_one_sum[3]+ vec_one_sum[4])/5
            vec_all_sum.append(vec_class_one)

        qry_nol_fuse = (outB_local_sum[0] + outB_local_sum[1])/2
        vec_all_fuse = (vec_all_sum[0] + vec_all_sum[1])/2
        vec_all_fuse = vec_all_fuse.unsqueeze(dim=2).unsqueeze(dim=3)
        outA_pred = []



        for i in range(2):
            outA_all_class = outA_all_sum[i]
            pred_A = []
            for j in range(5):
                outA_ima = outA_all_class[j]

                one_A_seg = self.cos_similarity_func(outA_ima, vec_all_fuse)
                outA_one_fuse = outA_ima * one_A_seg.unsqueeze(dim=1)
                outA_one_aspp = self.aspp(outA_one_fuse)  # 1*256*63*47
                outA_one_cla = self.classifier_6(outA_one_aspp)  # 1*128*63*47
                pred_one_A = self.cls_2way1shot(outA_one_cla)
                pred_A.append(pred_one_A)

            outA_pred.append(pred_A)


        tmp_seg = self.cos_similarity_func(qry_nol_fuse, vec_all_fuse)
        outB_fuse = qry_nol_fuse * tmp_seg.unsqueeze(dim=1)
        outB_aspp = self.aspp(outB_fuse)  # 1*256*63*47
        outB_cla = self.classifier_6(outB_aspp)  # 1*128*63*47
        outB_pred = self.cls_2way1shot(outB_cla)

        return outA_pred, outB_pred



    def forward_5shot_avg(self, anchor_img, pos_img_list, pos_mask_list):  # anchor_img=1*3*500*375; pos_img_list =

        outB = self.netB(anchor_img)  #1*256*63*47
        pred_sum = []
        pred_supp = []
        for i in range(5):
            pos_img = pos_img_list[i].cuda()
            pos_mask = pos_mask_list[i]
            pos_mask = self.warper_img(pos_mask)   #1*1*375*500
            outA_support = self.netB(pos_img)     #1*256*47*63
            a, b, c, d = outA_support.size()  # 1*256*48*63
            pos_maskdown = F.interpolate(pos_mask, size=(c, d))  # 1*1*47*63
            outA_down = outA_support * pos_maskdown  # #1*256*47*63
            vec_low = torch.sum(torch.sum(outA_down, dim=3), dim=2) / (torch.sum(pos_maskdown) + 1e-5)    #1*256
            _, _, mask_w, mask_h = pos_mask.size()
            outA_pos = F.upsample(outA_support, size=(mask_w, mask_h), mode='bilinear')                  #1*256*375*500
            vec_hig = torch.sum(torch.sum(outA_pos*pos_mask, dim=3), dim=2)/torch.sum(pos_mask)           #1*256
            vec_pos = 0.5*vec_low + 0.5*vec_hig                                                          #1*256
            w_b, w_c, w_k, w_g = outA_down.size()
            outA_down = (outA_down).view(w_b, w_c, w_g, w_k)    #1*256*63*47
            outB_nol = self.nol(outB, outA_down)                      #1*256*63*47

            vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)                                        #1*256*1*1

            tem_supp = self.cos_similarity_func(outA_support, vec_pos)
            outA_supp = outA_support * tem_supp.unsqueeze(dim=1)
            outA_fuse = self.aspp(outA_supp)  # 1*256*63*47
            outA_fuse = self.classifier_6(outA_fuse)  # 1*128*63*47
            outA_cls = self.cls(outA_fuse)
            pred_supp.append(outA_cls)

            tmp_seg = self.cos_similarity_func(outB_nol, vec_pos)                                       # 1*63*47
            outB_nol = outB_nol * tmp_seg.unsqueeze(dim=1)                                             # 1*256*63*47
            outB_fuse = self.aspp(outB_nol)                                                            # 1*256*63*47
            outB_fuse = self.classifier_6(outB_fuse)                                                   # 1*128*63*47
            outB_fuse = self.cls(outB_fuse)                                                            # 1*128*63*47
            pred_sum.append(outB_fuse)


        pred = (pred_sum[0] + pred_sum[1] +  pred_sum[2] + pred_sum[3] + pred_sum[4])/5


        return pred_supp, pred
    def forward_5shot_fuse(self, anchor_img, pos_img_list, pos_mask_list):  # anchor_img=1*3*500*375; pos_img_list =
        outB = self.netB(anchor_img)  #1*256*63*47
        outA_sum = []
        vec_sum = []
        outB_nol_sum = []
        pred_A_supp = []
        for i in range(5):
            pos_img = pos_img_list[i].cuda()
            pos_mask = pos_mask_list[i]
            pos_mask = self.warper_img(pos_mask)   #1*1*375*500
            outA_support = self.netB(pos_img)     #1*256*47*63
            outA_sum.append(outA_support)

            a, b, c, d = outA_support.size()  # 1*256*48*63
            pos_maskdown = F.interpolate(pos_mask, size=(c, d))  # 1*1*47*63
            outA_down = outA_support * pos_maskdown  # #1*256*47*63
            vec_low = torch.sum(torch.sum(outA_down, dim=3), dim=2) / (torch.sum(pos_maskdown) + 1e-5)    #1*256
            _, _, mask_w, mask_h = pos_mask.size()
            outA_pos = F.upsample(outA_support, size=(mask_w, mask_h), mode='bilinear')                  #1*256*375*500
            vec_hig = torch.sum(torch.sum(outA_pos*pos_mask, dim=3), dim=2)/torch.sum(pos_mask)           #1*256
            vec_pos = 0.5*vec_low + 0.5*vec_hig                                                          #1*256
            vec_sum.append(vec_pos)

            w_b, w_c, w_k, w_g = outA_down.size()
            outA_down = (outA_down).view(w_b, w_c, w_g, w_k)    #1*256*63*47
            outB_nol = self.nol(outB, outA_down)                      #1*256*63*47
            outB_nol_sum.append(outB_nol)

        vec_pos = (vec_sum[0]+vec_sum[1]+vec_sum[2]+vec_sum[3]+vec_sum[4])/5
        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)                                        #1*256*1*1
        for i in range(5):
            out_A_now = outA_sum[i]
            tem_supp = self.cos_similarity_func(out_A_now, vec_pos)
            outA_supp = out_A_now * tem_supp.unsqueeze(dim=1)
            outA_fuse = self.aspp(outA_supp)  # 1*256*63*47
            outA_fuse = self.classifier_6(outA_fuse)  # 1*128*63*47
            outA_cls = self.cls(outA_fuse)
            pred_A_supp.append(outA_cls)

        outB_nolocal = (outB_nol_sum[0]+outB_nol_sum[1]+outB_nol_sum[2]+outB_nol_sum[3]+outB_nol_sum[4])/5

        tmp_seg = self.cos_similarity_func(outB, vec_pos)                                                   # 1*63*47
        outB = outB * tmp_seg.unsqueeze(dim=1)                                                      # 1*256*63*47
        outB_fuse = self.aspp(outB)                                                                         # 1*256*63*47
        outB_fuse = self.classifier_6(outB_fuse)                                                                    # 1*128*63*47
        pred = self.cls(outB_fuse)                                                                             # 1*128*63*47

        return pred_A_supp, pred

    def forward_5shot_backnol(self, anchor_img, pos_img_list, pos_mask_list):  # anchor_img=1*3*500*375; pos_img_list =
        outB = self.netB(anchor_img)  #1*256*63*47
        outA_sum = []
        vec_sum = []
        outB_nol_sum = []
        pred_A_supp = []
        for i in range(5):
            pos_img = pos_img_list[i].cuda()
            pos_mask = pos_mask_list[i]
            pos_mask = self.warper_img(pos_mask)   #1*1*375*500
            outA_support = self.netB(pos_img)     #1*256*47*63
            outA_sum.append(outA_support)

            a, b, c, d = outA_support.size()  # 1*256*48*63
            pos_maskdown = F.interpolate(pos_mask, size=(c, d))  # 1*1*47*63
            outA_down = outA_support * pos_maskdown  # #1*256*47*63
            vec_low = torch.sum(torch.sum(outA_down, dim=3), dim=2) / (torch.sum(pos_maskdown) + 1e-5)    #1*256
            _, _, mask_w, mask_h = pos_mask.size()
            outA_pos = F.upsample(outA_support, size=(mask_w, mask_h), mode='bilinear')                  #1*256*375*500
            vec_hig = torch.sum(torch.sum(outA_pos*pos_mask, dim=3), dim=2)/torch.sum(pos_mask)           #1*256
            vec_pos = 0.5*vec_low + 0.5*vec_hig                                                          #1*256
            #vec_pos = vec_hig
            vec_sum.append(vec_pos)

            w_b, w_c, w_k, w_g = outA_down.size()
            outA_down = (outA_down).view(w_b, w_c, w_g, w_k)    #1*256*63*47
            outB_nol = self.nol(outB, outA_down)                      #1*256*63*47
            outB_nol_sum.append(outB_nol)

        vec_pos = (vec_sum[0]+vec_sum[1]+vec_sum[2]+vec_sum[3]+vec_sum[4])/5
        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)                                        #1*256*1*1
        for i in range(5):
            out_A_now = outA_sum[i]
            tem_supp = self.cos_similarity_func(out_A_now, vec_pos)
            outA_supp = out_A_now * tem_supp.unsqueeze(dim=1)
            outA_fuse = self.aspp(outA_supp)  # 1*256*63*47
            outA_fuse = self.classifier_6(outA_fuse)  # 1*128*63*47
            outA_cls = self.cls(outA_fuse)
            pred_A_supp.append(outA_cls)

        outB_nolocal = (outB_nol_sum[0]+outB_nol_sum[1]+outB_nol_sum[2]+outB_nol_sum[3]+outB_nol_sum[4])/5

        tmp_seg = self.cos_similarity_func(outB_nolocal, vec_pos)                                                   # 1*63*47
        outB = outB_nolocal * tmp_seg.unsqueeze(dim=1)                                                      # 1*256*63*47
        outB_fuse = self.aspp(outB)                                                                         # 1*256*63*47
        outB_fuse = self.classifier_6(outB_fuse)                                                                    # 1*128*63*47
        pred = self.cls(outB_fuse)                                                                             # 1*128*63*47

        return pred_A_supp, pred
    def warper_img(self, img):
        img_tensor = torch.Tensor(img).cuda()
        img_var = Variable(img_tensor)
        img_var = torch.unsqueeze(img_var, dim=0)
        return img_var

    def forward_5shot_max(self, anchor_img, pos_img_list, pos_mask_list):
        outB_side_list = []
        for i in range(5):
            pos_img = pos_img_list[i]
            pos_mask = pos_mask_list[i]

            pos_img = self.warper_img(pos_img)  #1*3*375*500
            pos_mask = self.warper_img(pos_mask)  #1*1*375*500

            outA_pos, _ = self.netB(pos_img)

            _, _, mask_w, mask_h = pos_mask.size()
            outA_pos = F.upsample(outA_pos, size=(mask_w, mask_h), mode='bilinear')
            vec_pos = torch.sum(torch.sum(outA_pos*pos_mask, dim=3), dim=2)/torch.sum(pos_mask)

            outB, outB_side = self.netB(anchor_img)

            # tmp_seg = outB * vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
            vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
            tmp_seg = self.cos_similarity_func(outB, vec_pos)

            exit_feat_in = outB_side * tmp_seg.unsqueeze(dim=1)
            outB_side_6 = self.classifier_6(exit_feat_in)
            outB_side = self.exit_layer(outB_side_6)

            outB_side_list.append(outB_side)

        return outB, outA_pos, vec_pos, outB_side_list

    def get_loss(self, logits, query_label):  #logits=1*2*43*67  ;query_label = 1*1*500*375
        out = logits

        b, c, w, h = query_label.size()
        outB_side = F.upsample(out, size=(w, h), mode='bilinear')  #1*2*500*375
        outB_side = outB_side.permute(0, 2, 3, 1).view(w * h, 2)
        query_label = query_label.view(-1)
        loss_bce_seg = self.bce_logits_func(outB_side, query_label.long())
        loss = loss_bce_seg

        return loss, 0, 0

    def get_2way_loss(self, logits, query_label):  #logits=1*2*43*67  ;query_label = 1*1*500*375
        out = logits

        b, c, w, h = query_label.size()
        outB_side = F.upsample(out, size=(w, h), mode='bilinear')  #1*2*500*375
        outB_side = outB_side.permute(0, 2, 3, 1).view(w * h, 3)
        query_label = query_label.view(-1)
        loss_bce_seg = self.bce_logits_func(outB_side, query_label.long())
        loss = loss_bce_seg

        return loss, 0, 0

    def get_pred_5shot_max(self, logits, query_label):
        outB, outA_pos, vec_pos, outB_side_list = logits

        w, h = query_label.size()[-2:]
        res_pred = None
        for i in range(5):
            outB_side = outB_side_list[i]
            outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')
            out_side = F.softmax(outB_side, dim=1).squeeze()
            values, pred = torch.max(out_side, dim=0)

            if res_pred is None:
                res_pred = pred
            else:
                res_pred = torch.max(pred, res_pred)

        return values, res_pred

    def get_pred(self, logits, query_image):
        out = logits
        w, h = query_image.size()[-2:]   #w=375, h =500
        out = F.upsample(out, size=(w, h), mode='bilinear')  #1*3*375*500
        out_softmax = F.softmax(out, dim=1).squeeze()   #3*375*500
        values, pred = torch.max(out_softmax, dim=0)  #375*500;   375*500

        return out_softmax, pred









# ------vgg----is backbone------------------

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from models.vgg import vgg_sg as vgg
# from oneshot.non_local import NONLocalBlock2D
# from oneshot.aspp_block import  _ASPP
#
#
# class OneModel(nn.Module):
#     def __init__(self, args):
#         super(OneModel, self).__init__()
#
#         self.netB = vgg.vgg16(pretrained=True, use_decoder=True)
#         # if args.isTest:
#         #     for p in self.parameters():
#         #         p.requires_grad = False
#
#         self.classifier_6 = nn.Sequential(
#             nn.Conv2d(512, 128, kernel_size=3, dilation=1,  padding=1),   #fc6
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 2, kernel_size=1, padding=0)
#         )
#         self.nol = NONLocalBlock2D(in_channels=512)
#
#         self.aspp = _ASPP(512, 512, [1, 6, 12, 18])
#
#         # self.dilated_one = nn.Sequential(self._make_layer(1, 512, 512, dilation=1, lastRelu=False), )
#         # self.dilated_two = nn.Sequential(self._make_layer(1, 512, 512, dilation=3, lastRelu=False), )
#         # self.dilated_three = nn.Sequential(self._make_layer(1, 512, 512, dilation=6, lastRelu=False), )
#         # self.dilated_four = nn.Sequential(self._make_layer(1, 512, 512, dilation=9, lastRelu=False), )
#
#
#         self.fuse = nn.Conv2d(2048, 512, kernel_size=1, padding=0)
#         # self.bce_logits_func = nn.BCEWithLogitsLoss()
#
#         self.bce_logits_func = nn.CrossEntropyLoss()
#         self.loss_func = nn.BCELoss()
#         self.cos_similarity_func = nn.CosineSimilarity()
#         self.triplelet_func = nn.TripletMarginLoss(margin=2.0)
#
#
#
#
#     def forward(self, anchor_img, pos_img, neg_img, pos_mask):      #query_label_var=1*3*375*500, pos_img=1*3*500*375, neg_img=1*1*500*375, pos_mask = 1*1*500*375
#
#         outA = self.netB(pos_img)     #1*512*47*63
#         outB = self.netB(anchor_img)                 #outB=1*512*47*63
#
#         a, b, c, d = outA.size()
#         pos_maskdown = F.interpolate(pos_mask, size=(c, d))
#         outA = outA * pos_maskdown
#         outB_nol = self.nol(outB, outA)
#
#
#
#         _, _, mask_w, mask_h = pos_mask.size() # pos_mask = 1*1*333*500
#         outA_pos = F.upsample(outA, size=(mask_w, mask_h), mode='bilinear')   ##1*512*500*375
#         vec_pos = torch.sum(torch.sum(outA_pos * pos_mask, dim=3), dim=2)/torch.sum(pos_mask)    #1*512
#         vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)   # 1*512*1*1
#         pos_seg = self.cos_similarity_func(outA, vec_pos)    # 1*63*47
#         outA = outA * pos_seg.unsqueeze(dim=1)              #1*512*63*47
#         outA = self.classifier_6(outA)                      # #1*2*63*47
#
#
#
#
#
#         tmp_seg = self.cos_similarity_func(outB_nol, vec_pos)     # 1*47*63
#         outB_nol = outB_nol * tmp_seg.unsqueeze(dim=1)             #1*512*42*63
#
#         outB_fuse = self.aspp(outB_nol)                             #1*512*42*63
#
#
#         outB_fuse = self.classifier_6(outB_fuse)         #1*2*48*63
#
#         return outA, outB_fuse
#
#
#
#
#
#
#     def forward_5shot_avg(self, anchor_img, pos_img_list, pos_mask_list):
#         vec_pos_sum = 0.
#         for i in range(5):
#             pos_img = pos_img_list[i]
#             pos_mask = pos_mask_list[i]
#
#             pos_img = self.warper_img(pos_img)
#             pos_mask = self.warper_img(pos_mask)
#
#             outA_pos, _ = self.netB(pos_img)
#
#             _, _, mask_w, mask_h = pos_mask.size()
#             outA_pos = F.upsample(outA_pos, size=(mask_w, mask_h), mode='bilinear')
#             vec_pos = torch.sum(torch.sum(outA_pos*pos_mask, dim=3), dim=2)/torch.sum(pos_mask)
#             vec_pos_sum += vec_pos
#
#         vec_pos = vec_pos_sum/5.0
#
#         outB, outB_side = self.netB(anchor_img)
#
#         # tmp_seg = outB * vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
#         vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
#         tmp_seg = self.cos_similarity_func(outB, vec_pos)
#
#         exit_feat_in = outB_side * tmp_seg.unsqueeze(dim=1)
#         outB_side_6 = self.classifier_6(exit_feat_in)
#         outB_side = self.exit_layer(outB_side_6)
#
#         return outB, outA_pos, vec_pos, outB_side
#
#     def warper_img(self, img):
#         img_tensor = torch.Tensor(img).cuda()
#         img_var = Variable(img_tensor)
#         img_var = torch.unsqueeze(img_var, dim=0)
#         return img_var
#
#     def forward_5shot_max(self, anchor_img, pos_img_list, pos_mask_list):
#         outB_side_list = []
#         for i in range(5):
#             pos_img = pos_img_list[i]
#             pos_mask = pos_mask_list[i]
#
#             pos_img = self.warper_img(pos_img)  #1*3*375*500
#             pos_mask = self.warper_img(pos_mask)  #1*1*375*500
#
#             outA_pos, _ = self.netB(pos_img)
#
#             _, _, mask_w, mask_h = pos_mask.size()
#             outA_pos = F.upsample(outA_pos, size=(mask_w, mask_h), mode='bilinear')
#             vec_pos = torch.sum(torch.sum(outA_pos*pos_mask, dim=3), dim=2)/torch.sum(pos_mask)
#
#             outB, outB_side = self.netB(anchor_img)
#
#             # tmp_seg = outB * vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
#             vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
#             tmp_seg = self.cos_similarity_func(outB, vec_pos)
#
#             exit_feat_in = outB_side * tmp_seg.unsqueeze(dim=1)
#             outB_side_6 = self.classifier_6(exit_feat_in)
#             outB_side = self.exit_layer(outB_side_6)
#
#             outB_side_list.append(outB_side)
#
#         return outB, outA_pos, vec_pos, outB_side_list
#
#     def get_loss(self, logits, query_label):  #logits=1*2*43*67  ;query_label = 1*1*500*375
#         out = logits
#
#         b, c, w, h = query_label.size()
#         outB_side = F.upsample(out, size=(w, h), mode='bilinear')  #1*2*500*375
#         outB_side = outB_side.permute(0, 2, 3, 1).view(w * h, 2)
#         query_label = query_label.view(-1)
#         loss_bce_seg = self.bce_logits_func(outB_side, query_label.long())
#         loss = loss_bce_seg
#
#         return loss, 0, 0
#
#
#     def get_pred_5shot_max(self, logits, query_label):
#         outB, outA_pos, vec_pos, outB_side_list = logits
#
#         w, h = query_label.size()[-2:]
#         res_pred = None
#         for i in range(5):
#             outB_side = outB_side_list[i]
#             outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')
#             out_side = F.softmax(outB_side, dim=1).squeeze()
#             values, pred = torch.max(out_side, dim=0)
#
#             if res_pred is None:
#                 res_pred = pred
#             else:
#                 res_pred = torch.max(pred, res_pred)
#
#         return values, res_pred
#
#     def get_pred(self, logits, query_image):
#         out = logits
#         w, h = query_image.size()[-2:]   #w=375, h =500
#         out = F.upsample(out, size=(w, h), mode='bilinear')  #1*2*375*500
#         out_softmax = F.softmax(out, dim=1).squeeze()   #2*375*500
#         values, pred = torch.max(out_softmax, dim=0)  #375*500;   375*500
#
#         return out_softmax, pred
#
#
