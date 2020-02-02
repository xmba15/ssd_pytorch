#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import torch.nn as nn


def build_loc_conf(
    num_classes,
    bbox_aspect_num=[4, 6, 6, 6, 4, 4],
    input_channels=[1024, 512, 512, 256, 256, 256],
):
    loc_net = []
    conf_net = []

    for i, (input_channel, aspect) in enumerate(
        zip(input_channels, bbox_aspect_num)
    ):
        loc_net.append(
            nn.Conv2d(input_channel, aspect * 4, kernel_size=3, padding=1,)
        )

        conf_net.append(
            nn.Conv2d(
                input_channel, aspect * num_classes, kernel_size=3, padding=1,
            )
        )

    return nn.ModuleList(loc_net), nn.ModuleList(conf_net)
