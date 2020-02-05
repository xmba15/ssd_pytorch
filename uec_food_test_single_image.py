#!/usr/bin/env python
# -*- coding: utf-8 -*-
from data_loader import UecFoodDatasetConfig, DataTransformBase
from test_process import test_process

if __name__ == "__main__":
    test_process(UecFoodDatasetConfig, DataTransformBase)
