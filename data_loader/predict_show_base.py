#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from .dataset_base import DatasetBase


def _normalize_fuc(img):
    img = img / 255.0
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img = (img - mean) / std

    return img


class PredictShowBase(object):
    def __init__(self, net, classes, colors, transform=None):
        self._net = net
        self._net.eval()

        self._classes = classes
        self._colors = colors
        self._transform = transform

    def __call__(self, img, conf_thresh=0.5, normalize_func=_normalize_fuc):
        orig_img = np.copy(img)
        height, width, _ = orig_img.shape

        if self._transform:
            img = self._transform(image=img)["image"]
        else:
            img = normalize_func(img)

        img = np.transpose(img, (2, 1, 0))
        if type(img) == np.ndarray:
            img = torch.from_numpy(img)
        input_x = img.unsqueeze(0)
        detections = self._net(input_x)

        all_bboxes = []
        all_category_ids = []
        scores = []

        detections = detections.cpu().detach().numpy()

        find_index = np.where(detections[:, 0:, :, 0] >= conf_thresh)
        detections = detections[find_index]
        for i in range(len(find_index[1])):
            if (find_index[1][i]) > 0:
                sc = detections[i][0]
                bbox = detections[i][1:] * [width, height, width, height]
                bbox = [int(elem) for elem in bbox]
                lable_ind = find_index[1][i] - 1

                all_bboxes.append(bbox)
                all_category_ids.append(lable_ind)
                scores.append(sc)

        result = DatasetBase.visualize_one_image_util(
            orig_img, self._classes, self._colors, all_bboxes, all_category_ids
        )

        return result
