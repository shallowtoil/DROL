# Copyright (c) SenseTime. All Rights Reserved.
# Modified by Jinghao Zhou

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch
from PIL import Image

from pysot.core.config import cfg
from pysot.utils.bbox import cxy_wh_2_rect
from pysot.tracker.classifier.libs.plotting import show_tensor
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from toolkit.utils.statistics import iou

class SiamMaskTracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamMaskTracker, self).__init__(model)
        assert hasattr(self.model, 'mask_head'), \
            "SiamMaskTracker must have mask_head"
        assert hasattr(self.model, 'refine_head'), \
            "SiamMaskTracker must have refine_head"

    def _crop_back(self, image, bbox, out_sz, padding=0):
        a = (out_sz[0] - 1) / bbox[2]
        b = (out_sz[1] - 1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _mask_post_processing(self, mask):
        target_mask = (mask > cfg.TRACK.MASK_THERSHOLD)
        target_mask = target_mask.astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]
            polygon = contour.reshape(-1, 2)
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))
            rbox_in_img = prbox
        else: 
            location = cxy_wh_2_rect(self.center_pos, self.size)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
        return rbox_in_img

    def track(self, img):
        
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        s_x = round(s_x)

        x_crop = self.get_subwindow(img,
                                    self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    s_x,
                                    self.channel_average)
        crop_box = [self.center_pos[0] - s_x / 2,
                    self.center_pos[1] - s_x / 2,
                    s_x,
                    s_x]

        with torch.no_grad():
            outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        def normalize(score):
            score = (score - np.min(score)) / (np.max(score) - np.min(score))
            return score

        if cfg.TRACK.USE_CLASSIFIER:
            flag, cs = self.classifier.track()
            confidence = Image.fromarray(cs.detach().cpu().numpy())
            confidence = np.array(confidence.resize((self.score_size, self.score_size))).flatten()
            pscore = pscore.reshape(5, -1) * (1 - cfg.TRACK.COEE_CLASS) + \
                normalize(confidence) * cfg.TRACK.COEE_CLASS
            pscore = pscore.flatten()

            if cfg.TRACK.TEMPLATE_UPDATE:
                score_st = self._convert_score(outputs['cls_st'])
                pred_bbox_st = self._convert_bbox(outputs['loc_st'], self.anchors)
                s_c_st = change(sz(pred_bbox_st[2, :], pred_bbox_st[3, :]) /
                                (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
                r_c_st = change((self.size[0] / self.size[1]) /
                                (pred_bbox_st[2, :] / pred_bbox_st[3, :]))
                penalty_st = np.exp(-(r_c_st * s_c_st - 1) * cfg.TRACK.PENALTY_K)
                pscore_st = penalty_st * score_st
                pscore_st = pscore_st.reshape(5, -1) * (1 - cfg.TRACK.COEE_CLASS) + \
                            normalize(confidence) * cfg.TRACK.COEE_CLASS
                pscore_st = pscore_st.flatten(0)

        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.TEMPLATE_UPDATE:
            pscore_st = pscore_st * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                     self.window * cfg.TRACK.WINDOW_INFLUENCE
            best_idx_st = np.argmax(pscore_st)
            bbox_st = pred_bbox_st[:, best_idx_st] / scale_z
            lr_st = penalty_st[best_idx_st] * score_st[best_idx_st] * cfg.TRACK.LR
            cx_st = bbox_st[0] + self.center_pos[0]
            cy_st = bbox_st[1] + self.center_pos[1]
            width_st = self.size[0] * (1 - lr_st) + bbox_st[2] * lr_st
            height_st = self.size[1] * (1 - lr_st) + bbox_st[3] * lr_st
            cx_st, cy_st, width_st, height_st = self._bbox_clip(cx_st, cy_st, width_st,
                                                    height_st, img.shape[:2])
            if iou((cx_st, cy_st, width_st, height_st), (cx, cy, width, height), wh=True) >= cfg.TRACK.TAU_REGRESSION \
                and score_st[best_idx_st] - score[best_idx] >= cfg.TRACK.TAU_CLASSIFICATION:
                cx, cy, width, height, score, best_idx = cx_st, cy_st, width_st, height_st, score_st, best_idx_st

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        pos = np.unravel_index(best_idx, (5, self.score_size, self.score_size))
        delta_x, delta_y = pos[2], pos[1]

        mask = self.model.mask_refine((delta_y, delta_x)).sigmoid().squeeze()
        out_size = cfg.TRACK.MASK_OUTPUT_SIZE
        mask = mask.view(out_size, out_size).cpu().data.numpy()

        s = crop_box[2] / cfg.TRACK.INSTANCE_SIZE
        base_size = cfg.TRACK.BASE_SIZE
        stride = cfg.ANCHOR.STRIDE
        sub_box = [crop_box[0] + (delta_x - base_size/2) * stride * s,
                   crop_box[1] + (delta_y - base_size/2) * stride * s,
                   s * cfg.TRACK.EXEMPLAR_SIZE,
                   s * cfg.TRACK.EXEMPLAR_SIZE]
        s = out_size / sub_box[2]

        im_h, im_w = img.shape[:2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, im_w*s, im_h*s]
        mask_in_img = self._crop_back(mask, back_box, (im_w, im_h))
        polygon = self._mask_post_processing(mask_in_img)
        polygon = polygon.flatten().tolist()

        if cfg.TRACK.USE_CLASSIFIER:
            self.classifier.update(bbox, scale_z, flag)

            if cfg.TRACK.TEMPLATE_UPDATE:
                if torch.max(cs).item() >= cfg.TRACK.TARGET_UPDATE_THRESHOLD and flag != 'hard_negative':
                    if torch.max(cs).item() > self.temp_max:
                        self.temp_max = torch.max(cs).item()
                        self.channel_average = np.mean(img, axis=(0, 1))
                        self.z_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.EXEMPLAR_SIZE,
                            s_z, self.channel_average)
                if (self.frame_num - 1) % cfg.TRACK.TARGET_UPDATE_SKIPPING == 0:
                    self.temp_max = 0
                    with torch.no_grad():
                        self.model.template_short_term(self.z_crop)

        return {
                'bbox': bbox,
                'best_score': best_score,
                'mask': mask_in_img,
                'polygon': polygon,
                'flag': flag
               }
