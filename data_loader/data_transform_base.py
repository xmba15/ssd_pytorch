#!/usr/bin/env python
# -*- coding: utf-8 -*-
from albumentations import BboxParams, Compose


class DataTransformBase(object):
    def __init__(self):
        self._train_transform_list = []
        self._val_transform_list = []

        self._transforms_dict = {}

    def _initialize_transform_dict(self):
        bbox_params = BboxParams(
            format="pascal_voc",
            min_area=0.001,
            min_visibility=0.001,
            label_fields=["category_ids"],
        )
        self._transforms_dict["train"] = Compose(
            self._train_transform_list, bbox_params=bbox_params
        )
        self._transforms_dict["val"] = Compose(
            self._val_transform_list, bbox_params=bbox_params
        )
        self._transforms_dict["test"] = Compose(self._val_transform_list)

    def __call__(self, image, bboxes=None, labels=None, phase=None):
        if phase is None:
            return self._transforms_dict["test"](image=image)

        assert phase in ("train", "val")
        assert bboxes is not None
        assert labels is not None

        annotations = {
            "image": image,
            "bboxes": bboxes,
            "category_ids": labels,
        }

        augmented = self._transforms_dict[phase](**annotations)

        return (
            augmented["image"],
            augmented["bboxes"],
            augmented["category_ids"],
        )
