#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .data_transform_base import DataTransformBase
from albumentations import (
    HorizontalFlip,
    GaussNoise,
    RandomBrightnessContrast,
)


class VOCDataTransform(DataTransformBase):
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
        super(VOCDataTransform, self).__init__(input_size, normalize)

        self._train_transform_list = self._train_transform_list + transforms

        self._val_transform_list = self._val_transform_list + []

        self._initialize_transform_dict()
