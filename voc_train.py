#!/usr/bin/env python
# -*- coding: utf-8 -*-
from train_process import train_process
from data_loader import VOCDataset, VOCDataTransform

if __name__ == "__main__":
    train_process(VOCDataset, VOCDataTransform)
