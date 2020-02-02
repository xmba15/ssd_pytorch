#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch
from .base_trainer import BaseTrainer


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))
try:
    from utils import inf_loop
except:
    print("cannot load modules")
    sys.exit(-1)


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        metric_func,
        optimizer,
        num_epochs,
        save_period,
        config,
        data_loaders_dict,
        scheduler=None,
        device=None,
        len_epoch=None,
    ):
        super(Trainer, self).__init__(
            model,
            criterion,
            metric_func,
            optimizer,
            num_epochs,
            save_period,
            config,
            device,
        )

        self.train_data_loader = data_loaders_dict["train"]
        self.val_data_loader = data_loaders_dict["val"]
        if len_epoch is None:
            self._len_epoch = len(self.train_data_loader)
        else:
            self.train_data_loader = inf_loop(self.train_data_loader)
            self._len_epoch = len_epoch

        self._do_validation = self.val_data_loader is not None
        self._scheduler = scheduler

    def _train_epoch(self, epoch):
        self._model.train()

        epoch_train_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data = data.to(self._device)
            target = [elem.to(self._device) for elem in target]
            self._optimizer.zero_grad()

            output = self._model(data)
            loss_c, loss_l = self._criterion(output, target)
            train_loss = loss_c + loss_l
            train_loss.backward()
            self._optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    "\n epoch: {} || iter: {} || loss_c: {} || loss_l: {} || total_loss: {}".format(
                        epoch,
                        batch_idx,
                        loss_c.item(),
                        loss_l.item(),
                        train_loss.item(),
                    )
                )

            epoch_train_loss += train_loss.item()
            if batch_idx == self._len_epoch:
                break

        if self._do_validation:
            epoch_val_loss = self._valid_epoch(epoch)

        if self._scheduler is not None:
            self._scheduler.step()

        epoch_train_loss /= len(self.train_data_loader)

        return epoch_train_loss, epoch_val_loss

    def _valid_epoch(self, epoch):
        print("start validation...")
        self._model.eval()

        epoch_val_loss = 0.0
        with torch.no_grad():
            for _, (data, target) in enumerate(self.val_data_loader):
                data = data.to(self._device)
                target = [elem.to(self._device) for elem in target]

                output = self._model(data)
                loss_c, loss_l = self._criterion(output, target)
                val_loss = loss_c + loss_l
                epoch_val_loss += val_loss.item()

        return epoch_val_loss / len(self.val_data_loader)
