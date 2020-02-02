#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch


def od_collate_fn(batch):
    imgs = []
    targets = []

    for sample in batch:
        img = sample[0]
        img = np.transpose(img, (2, 1, 0))
        img = np.expand_dims(img, axis=0)
        imgs.append(img)
        targets.append(torch.FloatTensor(sample[1]))

    imgs = np.concatenate(imgs, axis=0)
    imgs = torch.FloatTensor(imgs)

    return imgs, targets
