#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import torch
import torch.nn as nn
from .layers import VGG, build_extras, build_loc_conf, L2Norm
from .default_boxes import DBox
from .functions import Detect


class SSD300(nn.Module):
    __name__ = "ssd300"

    def __init__(self, phase, cfg, backbone=VGG()):
        super(SSD300, self).__init__()
        assert phase in ("train", "eval")
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        self.bbox_aspect_num = cfg["bbox_aspect_num"]
        self.top_k = cfg["top_k"]
        self.conf_thresh = cfg["conf_thresh"]
        self.nms_thresh = cfg["nms_thresh"]

        self.feature_extractor, out_channels = backbone
        self.L2Norm = L2Norm(512)
        self.additional_blocks = build_extras(
            input_size=out_channels
        )

        self.loc, self.conf = build_loc_conf(
            num_classes=self.num_classes,
            bbox_aspect_num=self.bbox_aspect_num,
        )

        self.dbox_list = torch.Tensor(DBox(cfg).build_dbox_list())

        if self.phase == "eval":
            self._detect = Detect(
                conf_thresh=self.conf_thresh,
                top_k=self.top_k,
                nms_thresh=self.nms_thresh,
            )

        self.feature_extractor.apply(self.weights_init)
        self.additional_blocks.apply(self.weights_init)
        self.loc.apply(self.weights_init)
        self.conf.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append(
                (
                    l(s).view(s.size(0), 4, -1),
                    c(s).view(s.size(0), self.num_classes, -1),
                )
            )

        locs, confs = list(zip(*ret))
        locs, confs = (
            torch.cat(locs, 2).permute(0, 2, 1).contiguous(),
            torch.cat(confs, 2).permute(0, 2, 1).contiguous(),
        )
        return locs, confs

    def forward(self, x):
        detection_feed = []

        for k in range(23):
            x = self.feature_extractor[k](x)
        source1 = self.L2Norm(x)
        detection_feed.append(source1)

        for k in range(23, len(self.feature_extractor)):
            x = self.feature_extractor[k](x)
        detection_feed.append(x)

        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        if self.phase == "eval":
            return self._detect(locs, confs, self.dbox_list)

        return locs, confs, self.dbox_list
