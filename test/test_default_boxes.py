#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import random
import numpy as np


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
try:
    sys.path.append(os.path.join(_CURRENT_DIR, ".."))
    from models import DBox
except:
    print("cannot load modules")
    exit(0)


def main():
    resize_size = (300, 300)
    ssd_cfg = {
        "input_size": resize_size[0],
        "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
        "feature_maps": [38, 19, 10, 5, 3, 1],
        "steps": [8, 16, 32, 64, 100, 300],
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        "top_k": 200,
        "conf_thresh": 0.5,
        "nms_thresh": 0.45,
    }
    dboxes = DBox(ssd_cfg, s_min=0.1).build_dbox_list()

    dboxes = dboxes * 300
    dboxes = np.array(dboxes, dtype=np.int64)
    dboxes = np.concatenate(
        (
            dboxes[:, :2] - dboxes[:, 2:] // 2,
            dboxes[:, :2] + dboxes[:, 2:] // 2,
        ),
        axis=-1,
    )
    dboxes = np.clip(dboxes, 0, 300)

    test_image_path = os.path.join(_CURRENT_DIR, "bird.png")
    test_image = cv2.imread(test_image_path)
    test_image = cv2.resize(test_image, resize_size)

    random.seed(100)
    indices = random.sample(range(len(dboxes)), 100)
    for i in indices:
        xmin, ymin, xmax, ymax = dboxes[i]
        cv2.rectangle(test_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

    cv2.imshow("default boxes", cv2.resize(test_image, (1000, 1000)))
    cv2.imwrite("default_boxes.png", cv2.resize(test_image, (1000, 1000)))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
