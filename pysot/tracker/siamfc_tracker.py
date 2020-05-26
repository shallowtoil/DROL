# Copyright (c) SenseTime. All Rights Reserved.
# Modified by Jinghao Zhou

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
from toolkit.utils.statistics import iou
import cv2

from pysot.core.config import cfg
from pysot.tracker.classifier.libs.plotting import show_tensor
from pysot.tracker.base_tracker import SiameseTracker
from pysot.tracker.classifier.base_classifier import BaseClassifier

class SiamFCTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamFCTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.TRACK.TOTAL_STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.interp_score_size = self.score_size * cfg.TRACK.RESPONSE_UP_STRIDE
        hanning = np.hanning(self.interp_score_size)[:, np.newaxis].dot(
            np.hanning(self.interp_score_size)[np.newaxis, :]).astype(np.float32)
        hanning /= np.sum(hanning)
        self.window = hanning
        self.lost_count = 0

        self.model = model
        self.model.eval()

        if cfg.TRACK.USE_CLASSIFIER:
            self.classifier = BaseClassifier(self.model)

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        self.frame_num = 1
        self.temp_max = 0

        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        self.channel_average = np.mean(img, axis=(0, 1))
        self.z0_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.EXEMPLAR_SIZE,
            s_z, self.channel_average)
        self.z_crop = self.z0_crop

        with torch.no_grad():
            self.model.template(self.z0_crop)

        if cfg.TRACK.USE_CLASSIFIER:
            if cfg.TRACK.TEMPLATE_UPDATE:
                with torch.no_grad():
                    self.model.template_short_term(self.z_crop)

            s_xx = s_z * (cfg.TRACK.INSTANCE_SIZE * 2 / cfg.TRACK.EXEMPLAR_SIZE)
            x_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.INSTANCE_SIZE * 2,
                round(s_xx), self.channel_average)
            self.classifier.initialize(x_crop.type(torch.FloatTensor), bbox)

        self.penalty = np.ones(cfg.TRACK.SCALE_NUM) * cfg.TRACK.PENALTY_K
        self.penalty[cfg.TRACK.SCALE_NUM//2] = 1
        self.scales = cfg.TRACK.SCALE_STEP ** np.arange(np.ceil(cfg.TRACK.SCALE_NUM/2) - cfg.TRACK.SCALE_NUM,
                                                  np.floor(cfg.TRACK.SCALE_NUM/2) + 1)

    def track(self, img):
        self.frame_num += 1
        self.curr_frame = img
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        size_x_scales = s_x * self.scales
        x_crop_pyramid = torch.cat([self.get_subwindow(img, self.center_pos, cfg.TRACK.INSTANCE_SIZE,
                        round(size_x_scale), self.channel_average) for size_x_scale in size_x_scales], dim=0)

        with torch.no_grad():
            outputs = self.model.track(x_crop_pyramid)

        score = outputs['cls'].squeeze()
        score = [cv2.resize(x.detach().cpu().numpy(), (self.interp_score_size,
                self.interp_score_size), cv2.INTER_CUBIC) for x in score]

        def normalize(score):
            score = (score - np.min(score)) / (np.max(score) - np.min(score))
            return score

        def normalize2(score):
            score -= score.min()
            score /= score.sum()
            return score

        max_score = np.array([x.max() for x in score])* self.penalty
        scale_idx = max_score.argmax()
        scale = self.scales[scale_idx]
        pscore = score[scale_idx]
        pscore = normalize2(pscore)

        if cfg.TRACK.USE_CLASSIFIER:

            flag, s = self.classifier.track(scale_idx)
            if flag == 'not_found':
                self.lost_count += 1
            else:
                self.lost_count = 0

            confidence = np.array(cv2.resize(s.detach().cpu().numpy(), (self.interp_score_size,
                self.interp_score_size), cv2.INTER_CUBIC))
            pscore = pscore * (1 - cfg.TRACK.COEE_CLASS) + \
                normalize2(confidence) * cfg.TRACK.COEE_CLASS

            if cfg.TRACK.TEMPLATE_UPDATE:
                score_st = outputs['cls_st'].squeeze()
                score_st = [cv2.resize(x.detach().cpu().numpy(), (self.interp_score_size,
                        self.interp_score_size), cv2.INTER_CUBIC) for x in score_st]
                max_score_st = np.array([x.max() for x in score_st]) * self.penalty
                scale_idx_st = max_score_st.argmax()
                scale_st = self.scales[scale_idx_st]
                pscore_st = score_st[scale_idx_st]
                pscore_st = normalize2(pscore_st)

        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        max_r, max_c = np.unravel_index(pscore.argmax(), pscore.shape)
        disp_score_interp = np.array([max_c, max_r]) - (self.interp_score_size - 1) / 2.
        disp_score_input = disp_score_interp * cfg.TRACK.TOTAL_STRIDE / cfg.TRACK.RESPONSE_UP_STRIDE
        disp_score_frame = disp_score_input * (s_x * scale) / cfg.TRACK.INSTANCE_SIZE

        if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.SHORT_TERM_DRIFT and self.lost_count >= 8:
            cx, cy = disp_score_frame[0] / 4 + self.center_pos[0], disp_score_frame[1] / 4 + self.center_pos[1]
        else:
            cx, cy = disp_score_frame[0] + self.center_pos[0], disp_score_frame[1] + self.center_pos[1]

        width = self.size[0] * ((1 - cfg.TRACK.LR) + cfg.TRACK.LR * scale)
        height = self.size[1] * ((1 - cfg.TRACK.LR) + cfg.TRACK.LR * scale)
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.TEMPLATE_UPDATE:

            pscore_st = pscore_st * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                     self.window * cfg.TRACK.WINDOW_INFLUENCE
            max_r_st, max_c_st = np.unravel_index(pscore_st.argmax(), pscore_st.shape)
            disp_score_interp_st = np.array([max_c_st, max_r_st]) - (self.interp_score_size - 1) / 2.
            disp_score_input_st = disp_score_interp_st * cfg.TRACK.TOTAL_STRIDE / cfg.TRACK.RESPONSE_UP_STRIDE
            disp_score_frame_st = disp_score_input_st * (s_x * scale) / cfg.TRACK.INSTANCE_SIZE
            if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.SHORT_TERM_DRIFT and self.lost_count >= 8:
                cx_st = disp_score_frame_st[0] / 4 + self.center_pos[0]
                cy_st = disp_score_frame_st[1] / 4 + self.center_pos[1]
            else:
                cx_st = disp_score_frame_st[0] + self.center_pos[0]
                cy_st = disp_score_frame_st[1] + self.center_pos[1]

            width_st = self.size[0] * ((1 - cfg.TRACK.LR) + cfg.TRACK.LR * scale_st)
            height_st = self.size[1] * ((1 - cfg.TRACK.LR) + cfg.TRACK.LR * scale_st)
            cx_st, cy_st, width_st, height_st = self._bbox_clip(cx_st, cy_st, width_st, height_st, img.shape[:2])
            if iou((cx_st, cy_st, width_st, height_st), (cx, cy, width, height), wh=True) >= cfg.TRACK.TAU_REGRESSION \
                and pscore_st.max() - pscore.max() >= cfg.TRACK.TAU_CLASSIFICATION:
                cx, cy, width, height, pscore= cx_st, cy_st, width_st, height_st, pscore_st

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2, cy - height / 2, width, height]
        best_score = pscore.max()

        if cfg.TRACK.USE_CLASSIFIER:
            self.classifier.update(bbox, flag)

            if cfg.TRACK.TEMPLATE_UPDATE:
                if torch.max(s).item() >= cfg.TRACK.TARGET_UPDATE_THRESHOLD and flag != 'hard_negative':
                    if torch.max(s).item() > self.temp_max:
                        self.temp_max = torch.max(s).item()
                        self.channel_average = np.mean(img, axis=(0, 1))
                        self.z_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.EXEMPLAR_SIZE,
                            s_z, self.channel_average)

                if (self.frame_num - 1) % cfg.TRACK.TARGET_UPDATE_SKIPPING == 0:
                    self.temp_max = 0
                    with torch.no_grad():
                        self.model.template_short_term(self.z_crop)

        if cfg.TRACK.USE_CLASSIFIER:
            return {
                    'bbox': bbox,
                    'best_score': best_score,
                    'flag': flag
                   }
        else:
            return {
                    'bbox': bbox,
                    'best_score': best_score
                   }
