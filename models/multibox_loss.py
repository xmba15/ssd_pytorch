#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .functions import match
from .default_boxes import DBox


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


class MultiBoxLoss(nn.Module):
    def __init__(
        self,
        alpha=1.0,
        jaccard_thresh=0.5,
        neg_pos=3,
        variances=[0.1, 0.2],
        device="cpu",
    ):
        super(MultiBoxLoss, self).__init__()
        self._alpha = alpha
        self._jaccard_thresh = jaccard_thresh
        self._neg_pos = neg_pos
        self._variances = variances
        self._device = device

    def forward(self, predictions, targets):
        loc_data, conf_data, dbox_list = predictions

        num_batch = loc_data.shape[0]
        num_dbox = loc_data.shape[1]
        num_classes = conf_data.shape[2]

        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self._device)
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self._device)
        dbox = dbox_list.to(self._device)

        for idx in range(num_batch):
            gt_bboxes = targets[idx][:, :-1].to(self._device)
            gt_labels = targets[idx][:, -1].to(self._device)

            match(
                self._jaccard_thresh,
                gt_bboxes,
                gt_labels,
                dbox,
                loc_t,
                conf_t_label,
                idx,
                self._variances,
            )

        loc_t = Variable(loc_t, requires_grad=False)
        conf_t_label = Variable(conf_t_label, requires_grad=False)

        pos_mask = conf_t_label > 0
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        batch_conf = conf_data.view(-1, num_classes)

        loss_c = F.cross_entropy(
            batch_conf, conf_t_label.view(-1), reduction="none"
        )

        num_pos = pos_mask.long().sum(1, keepdim=True)
        loss_c = loss_c.view(num_batch, -1)
        loss_c[pos_mask] = 0

        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        # hard negative mining
        num_neg = torch.clamp(num_pos * self._neg_pos, max=num_dbox)
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        conf_hnm = conf_data[(pos_idx_mask + neg_idx_mask).gt(0)].view(
            -1, num_classes
        )
        conf_t_label_hnm = conf_t_label[(pos_mask + neg_mask).gt(0)]

        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction="sum")

        N = num_pos.sum()
        loss_l /= N
        loss_c /= N

        return loss_c, loss_l
