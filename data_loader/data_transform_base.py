#!/usr/bin/env python
# -*- coding: utf-8 -*-
from albumentations import (
    BboxParams,
    Compose,
    Resize,
    Normalize,
    HorizontalFlip,
    GaussNoise,
    RandomBrightnessContrast,
)


class DataTransformBase(object):
    def __init__(
        self,
        transforms=[
            HorizontalFlip(p=0.5),
            GaussNoise(p=0.5),
            RandomBrightnessContrast(p=0.5),
        ],
        input_size=None,
        normalize=True,
    ):
        self._input_size = input_size
        self._normalize = normalize
        self._train_transform_list = transforms
        self._val_transform_list = []

        self._transforms_dict = {}
        self._bbox_params = BboxParams(
            format="pascal_voc",
            min_area=0.001,
            min_visibility=0.001,
            label_fields=["category_ids"],
        )
        self._initialize_transform_dict()

    def _initialize_transform_dict(self):
        if self._input_size is not None:
            height, width = self._input_size
            self._train_transform_list.append(
                Resize(height, width, always_apply=True)
            )
            self._val_transform_list.append(
                Resize(height, width, always_apply=True)
            )

        if self._normalize:
            self._train_transform_list.append(Normalize(always_apply=True))
            self._val_transform_list.append(Normalize(always_apply=True))

        self._transforms_dict["train"] = self._train_transform_list
        self._transforms_dict["val"] = self._val_transform_list
        self._transforms_dict["test"] = self._val_transform_list

    def __call__(self, image, bboxes=None, labels=None, phase=None):
        if phase is None:
            transformer = Compose(self._transforms_dict["test"])
            return transformer(image=image)

        assert phase in ("train", "val")
        assert bboxes is not None
        assert labels is not None

        transformed_image = image
        transformed_bboxes = bboxes
        transformed_category_ids = labels
        for transform in self._transforms_dict[phase]:
            annotations = {
                "image": transformed_image,
                "bboxes": transformed_bboxes,
                "category_ids": transformed_category_ids,
            }
            transformer = Compose([transform], bbox_params=self._bbox_params)
            augmented = transformer(**annotations)

            while len(augmented["bboxes"]) == 0:
                augmented = transformer(**annotations)

            transformed_image = augmented["image"]
            transformed_bboxes = augmented["bboxes"]
            transformed_category_ids = augmented["category_ids"]

        return (transformed_image, transformed_bboxes, transformed_category_ids)
