#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https: // github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
from numpy import random
import numpy as np
import cv2


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # [A,B]
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Expand(object):
    def __init__(self, mean=(104, 117, 123)):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth), dtype=image.dtype
        )
        expand_image[:, :, :] = self.mean
        expand_image[
            int(top) : int(top + height), int(left) : int(left + width)
        ] = image
        image = expand_image

        boxes = np.array(boxes)
        # print("00000000000000")
        # print(boxes)
        # boxes[:, :2]
        # print("00000000000000")

        boxes[:, :2] += [int(left), int(top)]
        boxes[:, 2:] += [int(left), int(top)]

        return image, boxes.tolist(), labels


class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes, labels):
        height, width, _ = image.shape
        boxes = np.array(boxes)
        labels = np.array(labels)

        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes.tolist(), labels.tolist()

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float("-inf")
            if max_iou is None:
                max_iou = float("inf")

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array(
                    [int(left), int(top), int(left + w), int(top + h)]
                )

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[
                    rect[1] : rect[3], rect[0] : rect[2], :
                ]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(
                    current_boxes[:, :2], rect[:2]
                )
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(
                    current_boxes[:, 2:], rect[2:]
                )
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return (
                    current_image,
                    current_boxes.tolist(),
                    current_labels.tolist(),
                )


class SSDAugmentation(object):
    def __init__(self):
        self.augment = Compose([Expand(), RandomSampleCrop(),])

    def __call__(self, img, boxes, labels, phase=None):
        return self.augment(img, boxes, labels)
