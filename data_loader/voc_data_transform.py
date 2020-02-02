#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .data_transform_base import DataTransformBase
from albumentations import (
    Resize,
    HorizontalFlip,
    Normalize,
    GaussNoise,
    RandomBrightnessContrast,
    RandomShadow,
    RandomRain,
    Rotate,
)
from albumentations.pytorch import ToTensor


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
        super(VOCDataTransform, self).__init__()

        self._train_transform_list = self._train_transform_list + transforms

        self._val_transform_list = self._val_transform_list + []

        if input_size is not None:
            height, width = input_size
            self._train_transform_list.append(
                Resize(height, width, always_apply=True)
            )
            self._val_transform_list.append(
                Resize(height, width, always_apply=True)
            )

        if normalize:
            self._train_transform_list.append(Normalize(always_apply=True))
            self._val_transform_list.append(Normalize(always_apply=True))

        self._initialize_transform_dict()
