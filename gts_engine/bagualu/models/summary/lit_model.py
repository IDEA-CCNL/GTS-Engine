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

from logging import Logger
from typing import Dict, List, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics.text.rouge import ROUGEScore
from transformers.optimization import (get_constant_schedule,
                                       get_linear_schedule_with_warmup)

from ...arguments.summary import TrainingArgumentsSummaryStd
from .eval_utils import SummarizationEvaluation
from .model import BagualuSummaryModel
from .tokenizers_pegasus import PegasusTokenizer


class BagualuSummaryLitModel(LightningModule):
    """BagualuSummaryLitModel.

    Args:
        args (TrainingArgumentsSummaryStd): arguments
        num_data (int, optional): num of input data. Defaults to 1.
    """

    _model: BagualuSummaryModel
    _logger: Logger

    def __init__(self,
                 args: TrainingArgumentsSummaryStd,
                 logger: Logger,
                 num_data: int = 1) -> None:
        super().__init__()

        self.args = args
        self.num_data = num_data
        self._model = BagualuSummaryModel(args.pretrained_model_root,
                                          args.max_dec_length)
        self._tokenizer = PegasusTokenizer.from_pretrained(
            self.args.pretrained_model_root)

        self._logger = logger

        self.num_step = -1
        self.num_step_each_epoch = -1

        self._summerization_evaluation = SummarizationEvaluation(
            self.args.pretrained_model_root)
        rouge_keys = tuple(args.rouge_keys.split(','))
        self._rouge_metric = ROUGEScore(rouge_keys=rouge_keys,
                                        normalizer=lambda x: x)

    def setup(self, stage: str) -> None:  # pylint: disable=signature-differs
        """set up.

        Args:
            stage (str): stage name
        """

        batch_size = self.args.batch_size
        num_devices = self.trainer.num_devices
        max_epochs = self.trainer.max_epochs
        accumulate_grad_batches = self.trainer.accumulate_grad_batches

        if stage == "fit":
            if self.trainer.max_steps and self.trainer.max_steps > 0:
                num_parallel = max(
                    1, num_devices if num_devices is not None else 0)
                num_step_each_epoch = int(self.num_data / num_parallel /
                                          accumulate_grad_batches / batch_size)
                self.num_step_each_epoch = num_step_each_epoch
                self.num_step = self.trainer.max_steps
                self._logger.info("----- set max steps -----")
                self._logger.info(f"num_data: {self.num_data}" )
                self._logger.info(f"num_parallel: {num_parallel}")
                self._logger.info(f"one epoch training step: {self.num_step_each_epoch}")
                self._logger.info(f"Total training step: {self.num_step}")
            elif max_epochs and max_epochs > 0:
                num_data = max_epochs * self.num_data
                num_parallel = max(
                    1, num_devices if num_devices is not None else 0)
                num_step = int(num_data / num_parallel /
                               accumulate_grad_batches / batch_size)
                self.num_step = num_step
                self.num_step_each_epoch = num_step / max_epochs
                self._logger.info("----- set max epochs -----")
                self._logger.info(f"num_data: {self.num_data}")
                self._logger.info(f"num_parallel: {num_parallel}")
                self._logger.info(f"one epoch training step: {self.num_step_each_epoch}")
                self._logger.info(f"Total training step: {self.num_step}")
            else:
                raise ValueError()

    def training_step(self, batch: dict, batch_idx: int):  # pylint: disable=signature-differs,arguments-differ,unused-argument
        """training.

        Args:
            batch (dict): batch of data
            batch_idx (int): batch index

        Returns:
            float: loss
        """
        model_output = self._model(**batch, is_training=True)

        loss = model_output['loss_total'].float()

        lr = self.lr_schedulers().get_last_lr()[0]
        self.log("train_step", self.global_step)
        self.log("lr", lr)
        self.log("train_loss", loss)

        if self.global_rank == 0:
            self._logger.info(
                f"loss: {loss:.4f} lr: {lr:.2e} epoch: {self.current_epoch} step: {self.global_step}"
            )

        return loss

    def validation_step(self, batch: dict, batch_idx: int):  # pylint: disable=signature-differs,arguments-differ,unused-argument
        """validation.

        Args:
            batch (dict): batch of data
            batch_idx (int): batch index
        """
        model_output = self._model(**batch, is_training=False)

        loss = model_output['loss_total'].float()
        generated_ids = model_output['generated_ids']

        preds, labels = self._summerization_evaluation.forward(
            generated_ids, batch['labels'])
        self._rouge_metric.update(preds=preds, target=labels)

        return [loss.item()]

    def validation_epoch_end(self, outputs) -> None:
        """Called at the end of the validation epoch with the outputs of all
        validation steps.

        Args:
            outputs(list): outputs of all val_step
        """
        loss_list = [step[0] for step in outputs]
        dev_loss = np.mean(loss_list)

        rouge_dict = self._rouge_metric.compute()
        self._rouge_metric.reset()

        metrics = {}
        for k, v in rouge_dict.items():
            metrics[f'dev_{k}'] = v

        metrics['dev_loss'] = dev_loss

        self.log_dict(metrics, prog_bar=False, logger=True, sync_dist=True)

        rougeL = rouge_dict[
            'rougeL_fmeasure'] if 'rougeL_fmeasure' in rouge_dict else 0.
        rouge1 = rouge_dict[
            'rouge1_fmeasure'] if 'rouge1_fmeasure' in rouge_dict else 0.
        rouge2 = rouge_dict[
            'rouge2_fmeasure'] if 'rouge2_fmeasure' in rouge_dict else 0.

        if self.global_rank == 0:
            self._logger.info(
                f"validation - loss: {dev_loss:.4f} rougeL_fmeasure: {rougeL:.4f}, rouge1_fmeasure: {rouge1:.4f}, rouge2_fmeasure: {rouge2:.4f} epoch: {self.current_epoch}"
            )

    def predict_step(self, batch: dict, batch_idx: int) -> List[Dict]:
        """predict.

        Args:
            batch (dict): batch of data
            batch_idx (int): batch index
        """
        model_output = self._model(**batch, is_training=False)

        preds = self._tokenizer.batch_decode(model_output['generated_ids'],
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)

        res = []
        for idx, true, pred in zip(batch['id'], batch['summary'], preds):
            res.append({'id': idx, 'summary': true, 'pred': pred})

        return res

    def configure_optimizers(self) -> List[dict]:
        """configure optimizers.

        Returns:
            List[dict]: optimizers
        """

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        paras = list(
            filter(lambda p: p[1].requires_grad, self.named_parameters()))

        paras = [{
            "params": [
                p for n, p in paras
                if not any(subname in n for subname in no_decay)
            ],
            "weight_decay":
            self.args.weight_decay
        }, {
            "params":
            [p for n, p in paras if any(subname in n for subname in no_decay)],
            "weight_decay":
            0.0
        }]

        optimizer = torch.optim.AdamW(paras, lr=self.args.learning_rate)

        if self.args.warmup > 0:
            num_warmup_steps = int(self.num_step * self.args.warmup)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps, self.num_step)
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
