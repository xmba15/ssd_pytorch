#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from .point_utils import center_size, point_form


def decode(loc, dbox_list, variances=[0.1, 0.2]):
    """
    dbox: (cx_d, cy_d, w_d, h_d)
    loc: (delta_x, delta_y,delta_w, delta_h)
    bbox: (cx, cy, w, h)
    cx = cx_d + variances[0] * delta_x * w_d
    cy = cy_d + variances[0] * delta_y * w_d
    w = w_d * exp(variances[1] * delta_w)
    h = h_d * exp(variances[1] * delta_w)

    Parameters
    ----------

    loc : [8732, 4]
    dbox_list : [8732, 4]

    Returns
    -------
    bboxes : (tensor) (x_min, x_max, y_min, y_max)
    """
    bboxes = torch.cat(
        (
            dbox_list[:, :2] + loc[:, :2] * variances[0] * dbox_list[:, 2:],
            dbox_list[:, 2:] * torch.exp(variances[1] * loc[:, 2:]),
        ),
        dim=-1,
    )

    return point_form(bboxes)


def encode(bboxes, dbox_list, variances=[0.1, 0.2]):
    """
    bboxes : (tensor) (x_min, x_max, y_min, y_max) (8732, 4)
    dbox_list : (tensor) (8732, 4)
    """
    center_points = center_size(bboxes)

    g_cxcy = (center_points[:, :2] - dbox_list[:, :2]) / (
        variances[0] * dbox_list[:, 2:]
    )
    g_wh = torch.log(center_points[:, 2:] / dbox_list[:, 2:]) / variances[1]

    # (num default box, 4)
    return torch.cat((g_cxcy, g_wh), dim=-1)
