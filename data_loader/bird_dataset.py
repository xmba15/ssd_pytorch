#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import random
from .dataset_base import DatasetBase, DatasetConfigBase


class BirdDatasetConfig(DatasetConfigBase):
    def __init__(self):
        super(BirdDatasetConfig, self).__init__()

        self.CLASSES = [
            "bird",
        ]

        self.COLORS = DatasetConfigBase.generate_color_chart(self.num_classes)


_bird_config = BirdDatasetConfig()


class BirdDataset(DatasetBase):
    __name__ = "bird_dataset"

    def __init__(
        self,
        data_path,
        classes=_bird_config.CLASSES,
        colors=_bird_config.COLORS,
        phase="train",
        transform=None,
        shuffle=True,
        random_seed=2000,
        normalize_bbox=False,
        bbox_transformer=None,
        train_val_ratio=0.9
    ):
        super(BirdDataset, self).__init__(
            data_path,
            classes=classes,
            colors=colors,
            phase=phase,
            transform=transform,
            shuffle=shuffle,
            normalize_bbox=normalize_bbox,
            bbox_transformer=bbox_transformer,
        )

        assert os.path.isdir(data_path)
        assert phase in ("train", "val", "test")

        self._data_path = os.path.join(data_path, "bird_dataset")
        assert os.path.isdir(self._data_path)

        self._phase = phase
        self._transform = transform

        if self._phase == "test":
            self._image_path_base = os.path.join(self._data_path, "test")
            self._image_paths = sorted(
                [
                    os.path.join(self._image_path_base, image_path)
                    for image_path in os.listdir(self._image_path_base)
                ]
            )
        else:
            self._image_path_base = os.path.join(self._data_path, "train_val")
            self._annotation_file = os.path.join(
                self._data_path, "annotations.csv"
            )
            lines = [
                line.rstrip("\n") for line in open(self._annotation_file, "r")
            ]
            lines = [line.split(",")[:-1] for line in lines]
            image_dict = {}
            for line in lines:
                if line[0] not in image_dict.keys():
                    image_dict[line[0]] = []
                image_dict[line[0]].append([int(e) for e in line[1:]])

            self._image_paths = image_dict.keys()

            self._image_paths = [
                os.path.join(self._data_path, elem)
                for elem in self._image_paths
            ]
            self._bboxes = image_dict.values()

            zipped = list(zip(self._image_paths, self._bboxes))
            random.seed(random_seed)
            random.shuffle(zipped)
            self._image_paths, self._bboxes = zip(*zipped)

            train_len = int(train_val_ratio * len(self._image_paths))
            if self._phase == "train":
                self._image_paths = self._image_paths[:train_len]
                self._bboxes = self._bboxes[:train_len]
            else:
                self._image_paths = self._image_paths[train_len:]
                self._bboxes = self._bboxes[train_len:]

            self._targets = [
                [img_bboxes, np.zeros(len(img_bboxes), dtype=np.int64)]
                for img_bboxes in self._bboxes
            ]

            del zipped
            del image_dict
