#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
from models import MultiBoxLoss, SSD300
from data_loader import UecFoodDataset, UecFoodDataTransform
from config import Config
from trainer import Trainer
from utils import od_collate_fn
from albumentations import (
    OneOf,
    Resize,
    Compose,
    CLAHE,
    HueSaturationValue,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    Normalize,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    HorizontalFlip,
    ShiftScaleRotate,
    Cutout,
    RandomSizedBBoxSafeCrop,
)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epoch", type=int, default=50)
parser.add_argument("--lr_rate", type=float, default=0.001)
parser.add_argument("--saved_period", default=10)
parser.add_argument("--snapshot", type=str)
parsed_args = parser.parse_args()


ssd_cfg = {
    "num_classes": 256 + 1,  # plus 1 for the background label
    "input_size": 300,
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300],
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    "top_k": 200,
    "conf_thresh": 0.5,
    "nms_thresh": 0.45,
}


def main(args):
    dt_config = Config()
    dt_config.display()
    input_size = (300, 300)

    transforms = [
        RandomSizedBBoxSafeCrop(height=400, width=400, p=0.5),
        OneOf([IAAAdditiveGaussianNoise(), GaussNoise(), ], p=0.5),
        GaussNoise(p=1.0),
        RandomContrast(limit=0.2, p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        RandomBrightness(limit=0.2, p=0.5),
        HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.5
        ),
        CLAHE(p=0.5, clip_limit=2.0),
        HorizontalFlip(p=0.5),
        Cutout(num_holes=5, p=0.5),
        ShiftScaleRotate(scale_limit=(-0.5, 0.1), p=0.5),
    ]

    data_transform = UecFoodDataTransform(
        transforms=transforms, input_size=input_size
    )
    train_dataset = UecFoodDataset(
        data_path=dt_config.DATA_PATH,
        phase="train",
        normalize_bbox=True,
        transform=data_transform,
    )

    val_dataset = UecFoodDataset(
        data_path=dt_config.DATA_PATH,
        phase="val",
        normalize_bbox=True,
        transform=data_transform,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=od_collate_fn,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=od_collate_fn,
    )
    data_loaders_dict = {"train": train_data_loader, "val": val_data_loader}

    model = SSD300(phase="train", cfg=ssd_cfg)
    # print(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = MultiBoxLoss(device=device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr_rate, weight_decay=2e-4
    )
    scheduler = lr_scheduler.StepLR(
        optimizer=optimizer, step_size=100, gamma=0.1
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        metric_func=None,
        optimizer=optimizer,
        num_epochs=args.num_epoch,
        save_period=args.saved_period,
        config=dt_config,
        data_loaders_dict=data_loaders_dict,
        scheduler=scheduler,
        device=device,
    )

    with torch.autograd.set_detect_anomaly(True):
        trainer.train()


if __name__ == "__main__":
    main(parsed_args)
