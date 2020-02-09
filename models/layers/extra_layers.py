#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn


def build_extras(input_size, channels=[256, 128, 128, 128]):
    additional_blocks = []
    for i, (input_size, output_size, channel) in enumerate(
        zip(input_size[:-1], input_size[1:], channels)
    ):
        if i < 2:
            layer = nn.Sequential(
                nn.Conv2d(input_size, channel, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    channel,
                    output_size,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
            )
        else:
            layer = nn.Sequential(
                nn.Conv2d(input_size, channel, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, output_size, kernel_size=3, bias=False),
                nn.ReLU(inplace=True),
            )

        additional_blocks.append(layer)

    return nn.ModuleList(additional_blocks)
