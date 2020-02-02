#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Function
from .decode_encode import decode
from .nms import nms


class Detect(Function):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        super(Detect, self).__init__()
        self._conf_thresh = conf_thresh
        self._top_k = top_k
        self._nms_thresh = nms_thresh
        self._softmax = nn.Softmax(dim=-1)

    def forward(self, loc_data, conf_data, dbox_list):
        batch_size = loc_data.shape[0]
        # num_dbox = loc_data.shape[1]
        num_classes = conf_data.shape[2]

        conf_data = self._softmax(conf_data)
        output = torch.zeros(batch_size, num_classes, self._top_k, 5)

        # conf_preds -> (batch_num, num_classes, num_dbox)
        conf_preds = conf_data.transpose(2, 1)

        for i in range(batch_size):
            # decoded_boxes->(num_dbox, 4)
            decoded_boxes = decode(loc_data[i], dbox_list)

            # conf_scores->(1, num_classes, num_dbox)
            conf_scores = conf_preds[i].clone()

            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(self._conf_thresh)

                scores = conf_scores[cl][c_mask]

                if scores.nelement() == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                bboxes = decoded_boxes[l_mask].view(-1, 4)

                ids, count = nms(bboxes, scores, self._nms_thresh, self._top_k)

                output[i, cl, :count] = torch.cat(
                    (scores[ids].unsqueeze(1), bboxes[ids]), dim=-1
                )

        return output
