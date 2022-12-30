# coding=utf-8
# Copyright 2021 The IDEA Authors. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=no-member

# import os
from typing import List, Tuple
from logging import Logger

import torch
from torch import Tensor
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule
from pytorch_lightning import LightningModule

from .model import BagualuIEModel
from ...arguments.ie import TrainingArgumentsIEStd
from ...lib.components.losses import NegativeSampleLoss, DiceLoss, DistillSelfLoss, DecoupledBCEloss


class BagualuIELitModel(LightningModule):
    """ BagualuIELitModel

    Args:
        args (TrainingArgumentsIEStd): arguments
        num_data (int, optional): num of input data. Defaults to 1.
    """

    _model: BagualuIEModel
    _logger: Logger

    def __init__(self, args: TrainingArgumentsIEStd, logger: Logger, num_data: int = 1) -> None:
        super().__init__()

        self.args = args
        self.num_data = num_data

        self._model = BagualuIEModel.from_pretrained(args.pretrained_model_root)
        self._logger = logger
        if args.loss == "dc":
            self.criterion = DiceLoss(smooth=args.dc_smooth,
                                      square_denominator=True,
                                      with_logits=False,
                                      ohem_ratio=0,
                                      alpha=0.,
                                      reduction="mean")
        elif args.loss == "bce":
            self.criterion = NegativeSampleLoss(args.neg_drop)
        elif args.loss == "d_bce":
            self.criterion = DecoupledBCEloss()
        else:
            raise NotImplementedError()

        self.num_step = -1
        self.num_step_each_epoch = -1
        self.distill_self = args.distill_self
        if self.distill_self:
            self.distill_self_loss = DistillSelfLoss()

    def setup(self, stage: str) -> None:  # pylint: disable=signature-differs
        """ set up

        Args:
            stage (str): stage name
        """

        batch_size = self.args.batch_size
        num_devices = self.trainer.num_devices
        max_epochs = self.trainer.max_epochs
        accumulate_grad_batches = self.trainer.accumulate_grad_batches

        if stage == "fit":
            if self.trainer.max_steps and self.trainer.max_steps > 0:
                num_parallel = max(1, num_devices if num_devices is not None else 0)
                num_step_each_epoch = int(self.num_data / num_parallel / accumulate_grad_batches / batch_size)
                self.num_step_each_epoch = num_step_each_epoch
                self.num_step = self.trainer.max_steps
                self._logger.info("----- set max steps -----")
                self._logger.info("num_data: %d", self.num_data)
                self._logger.info("num_parallel: %d", num_parallel)
                self._logger.info("one epoch training step: %d", self.num_step_each_epoch)
                self._logger.info("Total training step: %d", self.num_step)
            elif max_epochs and max_epochs > 0:
                num_data = max_epochs * self.num_data
                num_parallel = max(1, num_devices if num_devices is not None else 0)
                num_step = int(num_data / num_parallel / accumulate_grad_batches / batch_size)
                self.num_step = num_step
                self.num_step_each_epoch = num_step / max_epochs
                self._logger.info("----- set max epochs -----")
                self._logger.info("num_data: %d", self.num_data)
                self._logger.info("num_parallel: %d", num_parallel)
                self._logger.info("one epoch training step: %d", self.num_step_each_epoch)
                self._logger.info("Total training step: %d", self.num_step)
            else:
                raise ValueError()

    def training_step(self, batch: dict, batch_idx: int):  # pylint: disable=signature-differs,arguments-differ,unused-argument
        """ training

        Args:
            batch (dict): batch of data
            batch_idx (int): batch index

        Returns:
            float: loss
        """
        span_logits = self._model(**batch)
        span_labels = batch["span_labels"]
        span_mask = batch["label_mask"]

        loss = self.criterion(span_logits.reshape(-1),
                              span_labels.reshape(-1),
                              span_mask.reshape(-1)) * self.args.loss_boost

        if self.distill_self:
            mask_atten_logits, full_atten_logits = torch.chunk(span_logits, 2, dim=0)
            mask, _ = torch.chunk(span_mask, 2, dim=0)
            # label, label_1 = torch.chunk(span_labels, 2, dim=0)
            assert full_atten_logits.shape == mask_atten_logits.shape
            distill_loss1 = self.distill_self_loss(
                mask_atten_logits,
                full_atten_logits,
                mask)
            distill_loss2 = self.distill_self_loss(
                full_atten_logits,
                mask_atten_logits,
                mask)
            distill_loss = distill_loss1 + distill_loss2
            loss += distill_loss

        f1, recall, precise, sum_corr, sum_y_true, sum_y_pred = \
            self.compute_metrics(span_logits, span_labels, span_mask)

        self.log("train_step",  self.global_step)
        self.log("lr", self.lr_schedulers().get_last_lr()[0])
        self.log("train_loss", loss)
        self.log("train_f1", f1)
        self.log("train_recall", recall)
        self.log("train_precise", precise)
        # self.log("train_precise", precise, rank_zero_only=True)

        if self.global_rank == 0:
            if self.global_step % self.args.log_step == 0:
                if not self.distill_self:
                    self._logger.info("[TRAIN-%s] step:%s, epoch:%s, batch_idx:%s/%s, lr:%.2e, "
                                      "loss:%.6f, f1/p/r=%.4f/%.4f/%.4f, corr/pred/true=%d/%d/%d",
                                      self.global_rank,
                                      self.global_step,
                                      self.current_epoch,
                                      batch_idx,
                                      self.num_step_each_epoch,
                                      self.lr_schedulers().get_last_lr()[0],
                                      loss,
                                      f1,
                                      precise,
                                      recall,
                                      sum_corr,
                                      sum_y_pred,
                                      sum_y_true)
                else:
                    self._logger.info("[TRAIN-%s] step:%s, epoch:%s, batch_idx:%s/%s, lr:%.2e, "
                                      "loss:%.6f, mse loss:%.6f, f1/p/r=%.4f/%.4f/%.4f, "
                                      "corr/pred/true=%d/%d/%d",
                                      self.global_rank,
                                      self.global_step,
                                      self.current_epoch,
                                      batch_idx,
                                      self.num_step_each_epoch,
                                      self.lr_schedulers().get_last_lr()[0],
                                      loss,
                                      distill_loss,
                                      f1,
                                      precise,
                                      recall,
                                      sum_corr,
                                      sum_y_pred,
                                      sum_y_true)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:  # pylint: disable=signature-differs,arguments-differ,unused-argument
        """ validation

        Args:
            batch (dict): batch of data
            batch_idx (int): batch index
        """
        span_logits = self._model(**batch)
        span_labels = batch["span_labels"]
        span_mask = batch["label_mask"]

        loss = self.criterion(span_logits.reshape(-1),
                              span_labels.reshape(-1),
                              span_mask.reshape(-1)) * self.args.loss_boost

        _, _, _, sum_corr, sum_y_true, sum_y_pred = self.compute_metrics(span_logits, span_labels, span_mask)

        return [loss.cpu().detach().numpy(), sum_corr, sum_y_true, sum_y_pred]

    def validation_epoch_end(self, outputs):
        """ Called at the end of the validation epoch with the outputs of all validation steps.
        Args:
            outputs(list): outputs of all val_step
        """
        all_loss, all_acc, all_sum_y_pred, all_sum_y_true = [], 0., 0., 0.
        for loss, sum_corr, sum_y_true, sum_y_pred in outputs:
            all_loss.append(loss)
            all_acc += sum_corr
            all_sum_y_pred += sum_y_pred
            all_sum_y_true += sum_y_true

        recall = 0. if all_sum_y_true <= 0 else all_acc / all_sum_y_true
        precise = 0. if all_sum_y_pred <= 0 else all_acc / all_sum_y_pred
        f1 = 0. if recall + precise <= 0 else 2 * recall * precise / (recall + precise)

        loss = sum(all_loss) / len(all_loss)

        self._logger.info("\t[VALIDATION-%s] step:%s, epoch:%s, loss:%.6f, "
                          "f1/p/r=%.4f/%.4f/%.4f, corr/pred/true=%d/%d/%d",
                          self.global_rank,
                          self.global_step, self.current_epoch, loss,
                          f1,
                          precise,
                          recall,
                          all_acc,
                          all_sum_y_pred,
                          all_sum_y_true)

        self.log("val_loss", loss)
        self.log("val_f1", f1)

    def predict_step(self, batch: dict, batch_idx: int):  # pylint: disable=signature-differs,arguments-differ,unused-argument
        """ predict

        Args:
            batch (dict): batch of data
            batch_idx (int): batch index
        """
        span_logits = self._model(**batch)
        span_mask = batch["label_mask"]
        return span_logits, span_mask

    def configure_optimizers(self) -> List[dict]:
        """ configure optimizers

        Returns:
            List[dict]: optimizers
        """

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        paras = list(filter(lambda p: p[1].requires_grad, self.named_parameters()))

        paras = [
            {
                "params": [p for n, p in paras if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay
            },
            {
                "params": [p for n, p in paras if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]

        optimizer = torch.optim.AdamW(paras, lr=self.args.learning_rate)

        if self.args.warmup > 0:
            num_warmup_steps = int(self.num_step * self.args.warmup)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps,
                                                        self.num_step)
        else:
            scheduler = get_constant_schedule(optimizer)

        return [{
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }]

    def compute_metrics(self,
                        logits: Tensor,
                        labels: Tensor,
                        mask: Tensor) -> Tuple[float, float, float]:
        """ compute metrics

        Args:
            logits (Tensor): logits
            labels (Tensor): labels
            mask (Tensor): mask

        Returns:
            Tuple[float, float, float]: (f1, recall, precise)
        """
        ones = torch.ones_like(logits)
        zero = torch.zeros_like(logits)
        logits = torch.where(logits < 0.5, zero, ones)

        y_pred = logits.reshape(shape=(-1, ))
        y_true = labels.reshape(shape=(-1, ))
        mask = mask.reshape(shape=(-1, ))
        assert mask.shape == y_pred.shape == y_true.shape
        y_pred = y_pred * mask
        y_true = y_true * mask

        corr = torch.eq(y_pred, y_true).float()
        corr = torch.multiply(y_true, corr)

        sum_y_pred = torch.sum(y_pred.float())
        sum_y_true = torch.sum(y_true.float())
        sum_corr = torch.sum(corr.float())

        recall = 0. if sum_y_true <= 0 else sum_corr / sum_y_true
        precise = 0. if sum_y_pred <= 0 else sum_corr / sum_y_pred
        f1 = 0. if recall + precise <= 0 else 2 * recall * precise / (recall + precise)
        if sum_y_true <= 0 and sum_y_pred <= 0:
            f1, recall, precise = 1., 1., 1.

        return f1, recall, precise, sum_corr, sum_y_true, sum_y_pred
