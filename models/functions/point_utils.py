#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch


def center_size(bboxes):
    """
    Parameters:
    bboxes: (tensor) (x_min, y_min, x_max, y_max)

    Returns:
    bboxes: (tensor) (cx, cy, w, h)
    """
    return torch.cat(
        ((bboxes[:, 2:] + bboxes[:, :2]) / 2, bboxes[:, 2:] - bboxes[:, :2]),
        dim=-1,
    )


def point_form(bboxes):
    """
    Parameters:
    bboxes: (tensor) (cx, cy, w, h)

    Returns:
    bboxes: (tensor) (x_min, y_min, x_max, y_max)
    """
    return torch.cat(
        (bboxes[:, :2] - bboxes[:, 2:] / 2, bboxes[:, :2] + bboxes[:, 2:] / 2),
        dim=-1,
    )
