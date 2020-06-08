# Copyright (c) SenseTime. All Rights Reserved.
# Modified by Jinghao Zhou

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
from PIL import Image

from pysot.core.config import cfg
from pysot.tracker.classifier.libs.plotting import show_tensor
from pysot.tracker.siamrpn_tracker import SiamRPNTracker

class SiamRPNLTTracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamRPNLTTracker, self).__init__(model)
        self.longterm_state = False

    def track(self, img):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE
        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        score_size = (instance_size - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), self.anchor_num)
        anchors = self.generate_anchor(score_size)

        s_x = s_z * (instance_size / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos, instance_size,
                                    round(s_x), self.channel_average)
        with torch.no_grad():
            outputs = self.model.track(x_crop)
        
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        def normalize(score):
            score = (score - np.min(score)) / (np.max(score) - np.min(score))
            return score

        if cfg.TRACK.USE_CLASSIFIER:

            flag, s = self.classifier.track()
            confidence = Image.fromarray(s.detach().cpu().numpy())
            confidence = np.array(confidence.resize((score_size, score_size))).flatten()
            pscore = pscore.reshape(5, -1) * (1 - cfg.TRACK.COEE_CLASS) + \
                normalize(confidence) * cfg.TRACK.COEE_CLASS
            pscore = pscore.flatten()

        if not self.longterm_state:
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                    window * cfg.TRACK.WINDOW_INFLUENCE
        else:
            pscore = pscore * (1 - 0.001) + window * 0.001

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        best_score = score[best_idx]
        if best_score >= cfg.TRACK.CONFIDENCE_LOW:
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]
            width = self.size[0] * (1 - lr) + bbox[2] * lr
            height = self.size[1] * (1 - lr) + bbox[3] * lr
        else:
            cx = self.center_pos[0]
            cy = self.center_pos[1]
            width = self.size[0]
            height = self.size[1]

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])
        bbox = [cx - width / 2, cy - height / 2, width, height]

        if not self.longterm_state:
            if cfg.TRACK.USE_CLASSIFIER:
                self.classifier.update(bbox, scale_z, flag)

        if best_score < cfg.TRACK.CONFIDENCE_LOW:
            self.longterm_state = True
        elif best_score > cfg.TRACK.CONFIDENCE_HIGH:
            self.longterm_state = False

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

