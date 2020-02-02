#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from .point_utils import point_form
from .decode_encode import encode


def intersection_area(bboxes1, bboxes2):
    """
    Compute intersection area of two lists of bounding boxes in the point forms
    Parameters:
    bboxes1: (tensor) (number of boxes 1==n1, 4)
    bboxes2: (tensor) (number of boxes 2==n2, 4)

    Returns:
    matrix of intersection area: (tensor) (n1, n2)
    """
    n1 = bboxes1.shape[0]
    n2 = bboxes2.shape[0]

    max_xy = torch.min(
        bboxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
        bboxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2),
    )
    min_xy = torch.max(
        bboxes1[:, :2].unsqueeze(1).expand(n1, n2, 2),
        bboxes2[:, :2].unsqueeze(0).expand(n1, n2, 2),
    )

    intersection = torch.clamp((max_xy - min_xy), min=0)

    return intersection[:, :, 0] * intersection[:, :, 1]


def jaccard(bboxes1, bboxes2):
    """
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    """
    inter = intersection_area(bboxes1, bboxes2)

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])

    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    area1 = area1.unsqueeze(1).expand_as(inter)
    area2 = area2.unsqueeze(0).expand_as(inter)

    union = area1 + area2 - inter

    return inter / union


def match(
    threshold,
    gt_bboxes,
    gt_labels,
    dbox_list,
    loc_t,
    conf_t,
    idx,
    variances=[0.1, 0.2],
):
    # overlaps->(number of gt bboxes, number of default boxes)
    # for ssd300, number of default boxes = 8732
    overlaps = jaccard(gt_bboxes, point_form(dbox_list))

    # best default box for each ground truth
    _, best_prior_idx = overlaps.max(1, keepdim=True)

    # best ground truth for each default box
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    for j in range(best_prior_idx.shape[0]):
        best_truth_idx[best_prior_idx[j]] = j

    matches = gt_bboxes[best_truth_idx]

    conf = gt_labels[best_truth_idx] + 1
    conf[best_truth_overlap < threshold] = 0

    loc = encode(matches, dbox_list, variances)

    loc_t[idx] = loc
    conf_t[idx] = conf
