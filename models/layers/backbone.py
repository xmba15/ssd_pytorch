#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import torch.nn as nn
from torchvision.models.resnet import resnet50


class Resnet(nn.Module):
    def __init__(self, backbone_path=None):
        super().__init__()
        backbone = resnet50(pretrained=not backbone_path)
        self.out_channels = [1024, 512, 512, 256, 256, 256]
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x
