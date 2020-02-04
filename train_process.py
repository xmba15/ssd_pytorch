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
from data_loader import SSDAugmentation
from config import Config
from trainer import Trainer
from utils import od_collate_fn
from albumentations import (
    OneOf,
    CLAHE,
    HueSaturationValue,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    HorizontalFlip,
    ShiftScaleRotate,
    Cutout,
    RandomScale,
    IAAAffine,
    RandomShadow,
    MedianBlur,
    GaussianBlur,
    MotionBlur,
)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epoch", type=int, default=860)
parser.add_argument("--lr_rate", type=float, default=1e-3)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--gamma", default=0.1, type=float)
parser.add_argument("--milestones", default="571, 714", type=str)
parser.add_argument("--save_period", type=int, default=5)
parser.add_argument("--snapshot", type=str)
parsed_args = parser.parse_args()


def train_process(dataset_class, data_transform_class, input_size=(300, 300)):
    dt_config = Config()
    dt_config.display()

    transforms = [
        OneOf([IAAAdditiveGaussianNoise(), GaussNoise(),], p=0.5),
        OneOf([MedianBlur(), GaussianBlur(), MotionBlur()], p=0.5),
        RandomContrast(limit=0.2, p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        RandomBrightness(limit=0.2, p=0.5),
        HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=10, p=0.5
        ),
        CLAHE(clip_limit=2.0, p=0.5),
        HorizontalFlip(p=0.5),
        Cutout(num_holes=5, p=0.6),
        ShiftScaleRotate(scale_limit=0.5, rotate_limit=20, p=0.6),
        IAAAffine(p=0.5),
        RandomShadow(p=0.5),
    ]

    data_transform = data_transform_class(
        transforms=transforms, input_size=input_size
    )
    train_dataset = dataset_class(
        data_path=dt_config.DATA_PATH,
        phase="train",
        normalize_bbox=True,
        transform=[SSDAugmentation(), data_transform],
    )

    val_dataset = dataset_class(
        data_path=dt_config.DATA_PATH,
        phase="val",
        normalize_bbox=True,
        transform=data_transform,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=parsed_args.batch_size,
        shuffle=True,
        collate_fn=od_collate_fn,
        num_workers=4,
        drop_last=True,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=parsed_args.batch_size,
        shuffle=False,
        collate_fn=od_collate_fn,
        num_workers=4,
        drop_last=True,
    )
    data_loaders_dict = {"train": train_data_loader, "val": val_data_loader}

    ssd_cfg = {
        "num_classes": train_dataset.num_classes
        + 1,  # plus 1 for the background label
        "input_size": input_size[0],
        "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
        "feature_maps": [38, 19, 10, 5, 3, 1],
        "steps": [8, 16, 32, 64, 100, 300],
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        "top_k": 200,
        "conf_thresh": 0.5,
        "nms_thresh": 0.45,
    }

    model = SSD300(phase="train", cfg=ssd_cfg)
    # print(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = MultiBoxLoss(device=device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=parsed_args.lr_rate,
        momentum=parsed_args.momentum,
        weight_decay=parsed_args.weight_decay,
    )
    last_epoch = -1
    milestones = [int(v.strip()) for v in parsed_args.milestones.split(",")]
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1, last_epoch=last_epoch
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        metric_func=None,
        optimizer=optimizer,
        num_epochs=parsed_args.num_epoch,
        save_period=parsed_args.save_period,
        config=dt_config,
        data_loaders_dict=data_loaders_dict,
        scheduler=scheduler,
        device=device,
        dataset_name_base=train_dataset.__name__,
    )

    if parsed_args.snapshot and os.path.isfile(parsed_args.snapshot):
        trainer.resume_checkpoint(parsed_args.snapshot)

    with torch.autograd.set_detect_anomaly(True):
        trainer.train()
