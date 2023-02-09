from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Union

import torch
from bagualu.arguments.text_classification.arguments_std import (
    InferenceArgumentsClfStd, TrainingArgumentsClfStd)
from bagualu.models.text_classification.models_std import (
    InferenceModelClfStd, TrainingModelClfStd)
from gts_common.components.losses import compute_kl_loss
from gts_common.components.metrics import Logits2Acc
from gts_common.components.schedulers import \
    warmup_linear_decay_scheduler_factory
from gts_common.framework.classification_finetune import (
    BaseInferenceLightningClf, BaseTrainingLightningClf, StdLabel)
from gts_common.framework.classification_finetune.consts import (
    InfBatch, InferenceManagerOutput, InferenceModelOutput, TrainBatch)
from gts_common.framework.consts import BertInput
from gts_common.utils import LoggerManager
from torch import Tensor
from transformers.optimization import AdamW
from transformers.tokenization_utils import PreTrainedTokenizer


class TrainLightningClfStd(BaseTrainingLightningClf):

    #############################################################################################
    # setup
    #############################################################################################

    def __init__(
        self,
        args: TrainingArgumentsClfStd,
        class_num: int,
        sample_num: int,
    ):
        super().__init__()
        self._args = args
        self._model: TrainingModelClfStd = TrainingModelClfStd(
            self._args, self._args.pretrained_model_dir, class_num)
        self._sample_num = sample_num
        self._logits_2_acc = Logits2Acc()
        self._logger = LoggerManager.get_logger(self._args.logger)

    def configure_optimizers(self):
        optimizer = AdamW(self._model.parameters(),
                          lr=self._args.learning_rate,
                          eps=1e-8,
                          correct_bias=True)
        scheduler = warmup_linear_decay_scheduler_factory(
            optimizer=optimizer,
            warm_up_epoch=self._args.warm_up_epoch,
            decay_epoch=self._args.decay_epoch,
            epoch=self._args.max_epochs,
            train_data_length=self._sample_num,
            batch_size=self._args.train_batch_size)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def avg(self, loss_list, batch_size_list):
        return (loss_list * (batch_size_list / batch_size_list.sum())).sum()

    #############################################################################################
    # training
    #############################################################################################

    def training_step(self, batch: TrainBatch, batch_idx: int):
        bert_input = BertInput(input_ids=batch["input_ids"],
                               attention_mask=batch["input_mask"],
                               token_type_ids=batch["input_seg"],
                               labels=batch["labels"])
        batch_size = batch["input_ids"].shape[0]
        model_output = self._model.forward(bert_input,
                                           sample_weight=batch["weight"],
                                           label_id_clf=batch["label_id_clf"])
        loss = model_output["loss_total"].float()

        if self._args.use_rdrop:
            # rdrop
            logits = model_output["logits"]
            model_output = self._model.forward(
                bert_input,
                sample_weight=batch["weight"],
                label_id_clf=batch["label_id_clf"])
            loss2 = model_output["loss_total"].float()
            logits2 = model_output["logits"]
            kl_loss = compute_kl_loss(logits, logits2)
            loss = (loss + loss2) / 2 + self._args.rdrop_alpha * kl_loss
        lr = self.lr_schedulers().get_last_lr()[-1]  # type: ignore
        loss_ce = model_output["loss_ce"]
        loss_mlm = model_output["loss_mlm"]
        return {
            "loss": loss,
            "lr": lr,
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "loss_ce": loss_ce,
            "loss_mlm": loss_mlm
        }

    def training_step_end(self, step_output: Dict[str, Tensor]):
        """聚合training_step输出并打印."""
        lr = step_output["lr"]
        batch_idx = step_output["batch_idx"]
        loss = step_output["loss"]
        loss_mlm = step_output["loss_mlm"]
        loss_ce = step_output["loss_ce"]
        self._logger.info(
            f"loss: {loss:.4f} lr: {lr:.2e} batch: {batch_idx}/{self.trainer.num_training_batches} epoch: {self.current_epoch} step: {self.global_step}"
        )
        metrics = {
            "train_loss": loss,
            "lr": lr,
            "train_loss_ce": loss_ce,
            "train_loss_mlm": loss_mlm
        }
        self.log_dict(metrics)
        return loss

    #############################################################################################
    # validation
    #############################################################################################

    def on_validation_epoch_start(self) -> None:
        self._logger.info("validating...")

    def validation_step(self, batch: TrainBatch, batch_idx: int):
        bert_input = BertInput(input_ids=batch["input_ids"],
                               attention_mask=batch["input_mask"],
                               token_type_ids=batch["input_seg"],
                               labels=batch["labels"])
        model_output = self._model.forward(bert_input,
                                           sample_weight=batch["weight"],
                                           label_id_clf=batch["label_id_clf"],
                                           is_training=False)
        loss = model_output["loss_total"].float()
        logits = model_output["logits"]
        acc, _ = self._logits_2_acc.forward(logits, batch["label_id_clf"])
        batch_size = len(batch["input_ids"])
        return {"loss": loss, "acc": acc, "batch_size": batch_size}

    def validation_epoch_end(
            self, validation_step_outputs: List[Dict[str, Tensor]]) -> None:
        """聚合所有验证结果并打印."""
        loss_list = torch.stack(
            [step["loss"] for step in validation_step_outputs])
        acc_list = torch.stack(
            [step["acc"] for step in validation_step_outputs])
        batch_size_list = [
            step["batch_size"] for step in validation_step_outputs
        ]
        dev_loss = loss_list.mean()
        dev_acc = acc_list.sum() / sum(batch_size_list)
        metrics = {"dev_loss": dev_loss, "dev_acc": dev_acc}
        self.log_dict(metrics,
                      prog_bar=False,
                      logger=True,
                      rank_zero_only=True)
        self._logger.info(
            f"validation - loss: {dev_loss:.4f} acc: {dev_acc:.4f} epoch: {self.current_epoch}"
        )


class PredictLightningClfStd(BaseTrainingLightningClf):

    def __init__(
        self,
        label: StdLabel,
        args: TrainingArgumentsClfStd,
        tokenizer: PreTrainedTokenizer,
        datastore=None,
        best_hyper=None,
    ):
        super().__init__()
        self._model: InferenceModelClfStd = InferenceModelClfStd(
            label, args, tokenizer, datastore=datastore, best_hyper=best_hyper)
        self._label = label

    def forward(self, input_ids: Tensor, input_mask: Tensor,
                input_seg: Tensor):
        return InferenceModelOutput(
            **self._model.forward(input_ids, input_mask, input_seg))

    def predict_step(self,
                     batch: InfBatch,
                     batch_idx: int,
                     dataloader_idx: int = 0) -> List[Tuple[int, str]]:
        inference_output = self.forward(input_ids=batch["input_ids"],
                                        input_mask=batch["input_mask"],
                                        input_seg=batch["input_seg"])
        id_list: List[int] = batch["my_id"]
        prediction_id_list: List[int] = inference_output["positions"].squeeze(
        ).tolist()
        if not isinstance(prediction_id_list, list):  # 可能出现最后一个batch是单个值的情况
            prediction_id_list = [prediction_id_list]
        prediction_label_list = [
            self._label.id2label[prediction_id].label
            for prediction_id in prediction_id_list
        ]
        prediction_output_list = [
            (int(id), label)
            for id, label in zip(id_list, prediction_label_list)
        ]
        return prediction_output_list

    def on_predict_epoch_end(self, results) -> None:
        """聚合每个batch的预测结果."""
        concated_list: List[Tuple[int, str]] = sum(results[0], [])
        result_dict = {id: label for id, label in concated_list}
        results[0] = result_dict


class InferenceLightningClfStd(BaseInferenceLightningClf):

    def __init__(
        self,
        label: StdLabel,
        args: InferenceArgumentsClfStd,
        tokenizer: PreTrainedTokenizer,
        datastore=None,
        best_hyper=None,
    ):
        super().__init__()
        self._model: InferenceModelClfStd = InferenceModelClfStd(
            label, args, tokenizer, datastore=datastore,
            best_hyper=best_hyper)  # type: ignore
        self._label = label

    def forward(self, input_ids: Tensor, input_mask: Tensor,
                input_seg: Tensor):
        return InferenceModelOutput(
            **self._model.forward(input_ids, input_mask, input_seg))

    def predict_step(self,
                     batch: InfBatch,
                     batch_idx: int,
                     dataloader_idx: int = 0) -> InferenceManagerOutput:
        inference_output = self.forward(batch["input_ids"],
                                        batch["input_mask"],
                                        batch["input_seg"])
        prediction_id_list: List[int] = inference_output["positions"].squeeze(
        ).tolist()
        if not isinstance(prediction_id_list, list):  # 可能出现最后一个batch是单个值的情况
            prediction_id_list = [prediction_id_list]
        prediction_label_list = [
            self._label.id2label[prediction_id].label
            for prediction_id in prediction_id_list
        ]
        probabilities_list = inference_output["probs"].tolist()
        return InferenceManagerOutput(predictions=prediction_label_list,
                                      probabilities=probabilities_list)
