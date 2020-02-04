#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import torch
from data_loader import (
    PredictShowBase,
    VOCDatasetConfig,
    VOCDataTransform,
)
from models import SSD300


parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True)
parser.add_argument("--image_path", required=True)
parser.add_argument("--conf_thresh", type=float)
parsed_args = parser.parse_args()


ssd_cfg = {
    "num_classes": 20 + 1,  # plus 1 for the background label
    "input_size": 300,
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300],
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    "top_k": 200,
    "conf_thresh": 0.3,
    "nms_thresh": 0.45,
}


def main(args):
    assert args.snapshot and os.path.isfile(args.snapshot)
    assert args.image_path and os.path.isfile(args.image_path)

    input_size = (300, 300)
    data_transform = VOCDataTransform(input_size=input_size)
    data_config = VOCDatasetConfig()

    if args.conf_thresh:
        ssd_cfg["conf_thresh"] = args.conf_thresh
    conf_thresh = ssd_cfg["conf_thresh"]

    model = SSD300(phase="eval", cfg=ssd_cfg)
    model.load_state_dict(torch.load(args.snapshot)["state_dict"])

    predict_show_handler = PredictShowBase(
        model, data_config.CLASSES, data_config.COLORS, data_transform,
    )

    img = cv2.imread(args.image_path)
    result = predict_show_handler(img, conf_thresh=conf_thresh)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(parsed_args)
