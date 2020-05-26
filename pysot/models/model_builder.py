# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import torch
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.core.xcorr import xcorr_fast, xcorr_depthwise

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        if cfg.RPN.RPN:
            self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                         **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def avg(self, lst):
        return sum(lst) / len(lst)

    def weighted_avg(self, lst, weight):
        s = 0
        for i in range(len(weight)):
            s += lst[i] * weight[i]
        return s

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        if cfg.RPN.RPN:
            self.zf = zf
        else:
            self.zf = torch.cat([zf for _ in range(3)], dim=0)

    def template_short_term(self, z_st):
        zf_st = self.backbone(z_st)
        if cfg.MASK.MASK:
            zf_st = zf_st[-1]
        if cfg.ADJUST.ADJUST:
            zf_st = self.neck(zf_st)
        if cfg.RPN.RPN:
            self.zf_st = zf_st
        else:
            self.zf_st = torch.cat([zf_st for _ in range(3)], dim=0)

    def instance(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        if not cfg.ADJUST.LAYER:
            if cfg.ADJUST.FUSE == 'wavg':
                cls_weight = self.rpn_head.cls_weight
                self.cf = self.weighted_avg([cf for cf in xf], cls_weight)
            elif cfg.ADJUST.FUSE == 'avg':
                self.cf = self.avg([cf for cf in xf])
            elif cfg.ADJUST.FUSE == 'con':
                self.cf = torch.cat([cf for cf in xf], dim=1)
        else:
            if isinstance(xf, list):
                self.cf = xf[cfg.ADJUST.LAYER-1]
            else:
                self.cf = xf

    def track(self, x):

        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        if not cfg.ADJUST.LAYER:
            if cfg.ADJUST.FUSE == 'wavg':
                cls_weight = self.rpn_head.cls_weight
                self.cf = self.weighted_avg([cf for cf in xf], cls_weight)
            elif cfg.ADJUST.FUSE == 'avg':
                self.cf = self.avg([cf for cf in xf])
            elif cfg.ADJUST.FUSE == 'con':
                self.cf = torch.cat([cf for cf in xf], dim=1)
        else:
            if isinstance(xf, list):
                self.cf = xf[cfg.ADJUST.LAYER-1]
            else:
                self.cf = xf

        if cfg.RPN.RPN:
            cls, loc = self.rpn_head(self.zf, xf)
        else:
            b, _, h, w = xf.size()
            cls = F.conv2d(xf.view(1, -1, h, w), self.zf, groups=b).transpose(0, 1)

        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)

        if cfg.TRACK.TEMPLATE_UPDATE:
            if cfg.RPN.RPN:
                cls_st, loc_st = self.rpn_head(self.zf_st, xf)
            else:
                b, _, h, w = xf.size()
                cls_st = F.conv2d(xf.view(1, -1, h, w), self.zf_st, groups=b).transpose(0,1)
            return {
                    'cls': cls,
                    'loc': loc if cfg.RPN.RPN else None,
                    'cls_st': cls_st,
                    'loc_st': loc_st if cfg.RPN.RPN else None,
                    'mask': mask if cfg.MASK.MASK else None
                   }
        else:
            return {
                'cls': cls,
                'loc': loc if cfg.RPN.RPN else None,
                'mask': mask if cfg.MASK.MASK else None
            }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        if cfg.RPN.RPN:
            cls, loc = self.rpn_head(zf, xf)

            # get loss
            cls = self.log_softmax(cls)
            cls_loss = select_cross_entropy_loss(cls, label_cls)
            loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

            outputs = {}
            outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                cfg.TRAIN.LOC_WEIGHT * loc_loss
            outputs['cls_loss'] = cls_loss
            outputs['loc_loss'] = loc_loss
        else:
            b, _, h, w = xf.size()
            cls = F.conv2d(xf.view(1, -1, h, w), zf, groups=b) * 1e-3 + self.backbone.corr_bias
            cls = cls.transpose(0, 1)

            # get loss
            cls = self.log_softmax(cls)
            cls_loss = select_cross_entropy_loss(cls, label_cls)

            outputs = {}
            outputs['total_loss'] = cls_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs
