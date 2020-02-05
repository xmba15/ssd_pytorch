#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _one_hot(index, num_classes, device):
    size = index.size() + (num_classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.0

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1)).to(device)
        mask = Variable(mask, requires_grad=index.requires_grad).to(device)

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):
    def __init__(self, device, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self._gamma = gamma
        self._eps = eps
        self._device = device

    def forward(self, predictions, targets):
        y = _one_hot(targets, predictions.size(-1), self._device)
        logit = F.softmax(predictions, dim=-1)
        logit = logit.clamp(self._eps, 1.0 - self._eps)

        loss = -1 * y * torch.log(logit)
        loss = loss * (1 - logit) ** self._gamma

        return loss.sum()
