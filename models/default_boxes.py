#!/usr/bin/env python
# -*- coding: utf-8 -*-
from itertools import product
import numpy as np


class DBox(object):
    """
    Read [the paper](https://arxiv.org/pdf/1512.02325.pdf) for more detail
    """

    def __init__(self, cfg, s_min=0.2, s_max=0.9):
        super(DBox, self).__init__()
        self._image_size = cfg["input_size"]
        self._feature_maps = cfg["feature_maps"]
        self._num_priors = len(self._feature_maps)
        assert self._num_priors > 0
        self._steps = cfg["steps"]
        self._s_min = s_min
        self._s_max = s_max
        self._sks = self._estimate_sks(
            self._num_priors, self._s_min, self._s_max
        )
        self._aspect_ratios = cfg["aspect_ratios"]

    def build_dbox_list(self):
        mean = []
        for k, feature_size in enumerate(self._feature_maps):
            for i, j in product(range(feature_size), repeat=2):
                f_k = self._image_size / self._steps[k]
                c_x = (j + 0.5) / f_k
                c_y = (i + 0.5) / f_k

                s_k = self._sks[k]
                s_k_next = self._sks[k + 1]

                mean += [c_x, c_y, s_k, s_k]
                s_k_prime = np.sqrt(s_k * s_k_next)
                mean += [c_x, c_y, s_k_prime, s_k_prime]

                for aspect_ratio in self._aspect_ratios[k]:
                    mean += [
                        c_x,
                        c_y,
                        s_k * np.sqrt(aspect_ratio),
                        s_k / np.sqrt(aspect_ratio),
                    ]
                    mean += [
                        c_x,
                        c_y,
                        s_k / np.sqrt(aspect_ratio),
                        s_k * np.sqrt(aspect_ratio),
                    ]

        mean = np.array(mean).reshape(-1, 4)
        mean = np.clip(mean, 0.0, 1.0)

        return mean

    def _estimate_sks(self, m=6, s_min=0.2, s_max=0.9):
        """
        m: number of feature maps
        s_min: minimum scale
        s_max: maximum scale
        """

        return [
            s_min + (s_max - s_min) * (k - 1) / (m - 1) for k in range(1, m + 2)
        ]
