# Copyright (c) SenseTime. All Rights Reserved.
# Modified by Jinghao Zhou

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamfc_tracker import SiamFCTracker
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker
from pysot.tracker.siammask_tracker import SiamMaskTracker

TRACKS = {
          'SiamFCTracker': SiamFCTracker,
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
