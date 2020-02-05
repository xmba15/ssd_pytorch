#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import torch
from data_loader import PredictShowBase
from models import SSD300


parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True)
parser.add_argument("--image_path", required=True)
parser.add_argument("--conf_thresh", type=float)
parser.add_argument("--output_size", type=int)
parsed_args = parser.parse_args()


def test_process(
    dataset_config_class, data_transform_class, input_size=(300, 300)
):
    assert parsed_args.snapshot and os.path.isfile(parsed_args.snapshot)
    assert parsed_args.image_path and os.path.isfile(parsed_args.image_path)

    data_transform = data_transform_class(input_size=input_size)
    data_config = dataset_config_class()

    ssd_cfg = {
        # plus 1 for the background label
        "num_classes": data_config.num_classes + 1,
        "input_size": 300,
        "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
        "feature_maps": [38, 19, 10, 5, 3, 1],
        "steps": [8, 16, 32, 64, 100, 300],
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        "top_k": 200,
        "conf_thresh": 0.3,
        "nms_thresh": 0.45,
    }

    if parsed_args.conf_thresh:
        ssd_cfg["conf_thresh"] = parsed_args.conf_thresh
    conf_thresh = ssd_cfg["conf_thresh"]

    model = SSD300(phase="eval", cfg=ssd_cfg)
    model.load_state_dict(torch.load(parsed_args.snapshot)["state_dict"])

    predict_show_handler = PredictShowBase(
        model, data_config.CLASSES, data_config.COLORS, data_transform,
    )

    img = cv2.imread(parsed_args.image_path)
    result = predict_show_handler(img, conf_thresh=conf_thresh)

    if parsed_args.output_size:
        result = cv2.resize(
            result, (parsed_args.output_size, parsed_args.output_size)
        )

    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
