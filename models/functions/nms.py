#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch


def nms(bboxes, scores, overlap=0.45, top_k=200):
    """
    Parameters
    ----------

    bboxes: bbox coordinates of shape[number of boxes, 4]
    scores: confidences of shape[number of boxes]

    Returns
    -------

    keep: list of index of bboxes to keep in the order of confidences
    count: number of kept bboxes
    """
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()

    x_min = bboxes[:, 0]
    y_min = bboxes[:, 1]
    x_max = bboxes[:, 2]
    y_max = bboxes[:, 3]
    area = torch.mul(x_max - x_min, y_max)

    _, indices = scores.sort(0)
    indices = indices[-top_k:]

    while indices.numel() > 0:
        cur_idx = indices[-1]
        keep[count] = cur_idx
        count += 1

        if indices.size(0) == 1:
            break

        indices = indices[:-1]

        selected_bboxes = bboxes[indices]

        tmp_x_min = selected_bboxes[:, 0]
        tmp_y_min = selected_bboxes[:, 1]
        tmp_x_max = selected_bboxes[:, 2]
        tmp_y_max = selected_bboxes[:, 3]

        tmp_x_min = torch.clamp(tmp_x_min, min=x_min[cur_idx])
        tmp_y_min = torch.clamp(tmp_y_min, min=y_min[cur_idx])
        tmp_x_max = torch.clamp(tmp_x_max, max=x_max[cur_idx])
        tmp_y_max = torch.clamp(tmp_y_max, max=y_max[cur_idx])

        tmp_w = tmp_x_max - tmp_x_min
        tmp_h = tmp_y_max - tmp_y_min

        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        intersection_area = tmp_w * tmp_h
        rem_area = torch.index_select(area, 0, indices)
        union = (rem_area - intersection_area) + area[cur_idx]
        IoU = intersection_area / union

        indices = indices[IoU.le(overlap)]

    return keep[:count], count
