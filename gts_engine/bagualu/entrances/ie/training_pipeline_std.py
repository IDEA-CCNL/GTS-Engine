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

import os
import datetime
import shutil
from logging import Logger
from typing import List, Optional

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ...arguments.ie import TrainingArgumentsIEStd
from ...models.ie import BagualuIELitModel, BagualuIEExtractModel, BagualuIEOnnxConfig
from ...lib.framework.base_training_pipeline import BaseTrainingPipeline
from ...lib.utils import LoggerManager
from ...lib.utils.json_processor import (load_json_list,
                                         dump_json,
                                         dump_json_list)
from ...dataloaders.ie import (BagualuIEDataModel,
                               check_data,
                               data_segment,
                               data_segment_restore,
                               eval_entity_f1,
                               eval_entity_relation_f1)


class TrainingPipelineIEStd(BaseTrainingPipeline):
    """ training pipeline """

    _args: TrainingArgumentsIEStd # training所需参数

    def __init__(self, args_parse_list: Optional[List[str]] = None):
        super().__init__(args_parse_list)

        self._logger: Logger
        self._tokenizer: PreTrainedTokenizer
        self._data_module: BagualuIEDataModel
        self._lit_model: BagualuIELitModel
        self._trainer: Trainer
        self._extract_model: BagualuIEExtractModel

    def _before_training(self) -> None:
        self._set_logger()
        self._logger.info("use saving path: %s", self._args.ft_output_dir)
        self._generate_tokenizer()
        self._logger.info("phase before_training finished")

    def _train(self) -> None:
        self._get_data_module()
        self._get_training_lightning()
        self._get_trainer()
        self._fit()
        self._logger.info("phase train finished")

    def _after_training(self) -> None:
        best_ckpt = self._select_best_model_from_ckpts()
        self._get_inf_model(best_ckpt)
        self._extract_model = BagualuIEExtractModel(self._tokenizer, self._args)
        self._generate_prediction_file()
        self._export_onnx()
        self._logger.info("phase after_training finished")

    def _set_logger(self) -> None:
        now_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        task_name = f"{self._args.dataset}_{now_date}"
        log_file = os.path.join(self._args.log_dir, task_name + ".log")
        LoggerManager.set_logger(self._args.logger, log_file)
        self._logger = LoggerManager.get_logger(self._args.logger)
        self._logger.info("log_file: %s", log_file)

    def _generate_tokenizer(self):
        self._logger.info("generate tokenizer...")
        added_token = [f"[unused{i + 1}]" for i in range(99)]
        tokenizer = AutoTokenizer.from_pretrained(self._args.pretrained_model_root,
                                                  additional_special_tokens=added_token)
        self._tokenizer = tokenizer

    def _get_data_module(self):
        train_data_path = self._args.train_data_path
        dev_data_path = self._args.dev_data_path
        test_data_path = self._args.test_data_path

        # train data
        if train_data_path and os.path.exists(train_data_path):
            train_data = list(load_json_list(train_data_path))
            self._logger.info("load train data from %s", train_data_path)
            check_data(train_data)
            train_data = data_segment(train_data)
        else:
            train_data = None
            self._logger.error("training data path [%s] is not valid", train_data_path)

        # dev data
        if dev_data_path and os.path.exists(dev_data_path):
            dev_data = list(load_json_list(dev_data_path))
            self._logger.info("load dev data from %s", dev_data_path)
            check_data(dev_data)
            dev_data = data_segment(dev_data)
        else:
            dev_data = None
            self._logger.warning("dev data path [%s] is not valid", dev_data_path)

        # test data
        if test_data_path and os.path.exists(test_data_path):
            test_data = list(load_json_list(test_data_path))
            self._logger.info("load test data from %s", test_data_path)
            check_data(test_data)
            test_data = data_segment(test_data)
        else:
            test_data = None
            self._logger.warning("test data path [%s] is not valid", test_data_path)

        self._data_module =  BagualuIEDataModel(self._tokenizer,
                                                self._args,
                                                train_data=train_data,
                                                dev_data=dev_data,
                                                test_data=test_data)

    def _get_training_lightning(self):
        self._lit_model = BagualuIELitModel(self._args, logger=self._logger)

    def _get_trainer(self):
        # add checkpoint callback
        checkpoint_callback = ModelCheckpoint(monitor=self._args.ckpt_monitor,
                                              save_top_k=self._args.ckpt_save_top_k,
                                              mode=self._args.ckpt_mode,
                                              save_last=self._args.ckpt_save_last,
                                              every_n_train_steps=self._args.ckpt_every_n_train_steps,
                                              save_weights_only=self._args.ckpt_save_weights_only,
                                              dirpath=self._args.ckpt_dirpath,
                                              filename=self._args.ckpt_filename)

        # add early stop callback
        early_stop_callback = EarlyStopping(monitor=self._args.ckpt_monitor,
                                            mode=self._args.ckpt_mode,
                                            patience=10)

        # add logger
        logger = TensorBoardLogger(save_dir=self._args.tensorboard_dir)

        # trainer
        trainer = Trainer(logger=logger,
                          precision=self._args.precision,
                          callbacks=[checkpoint_callback, early_stop_callback],
                          accelerator=self._args.accelerator,
                          devices=self._args.gpus,
                          max_steps=self._args.max_steps,
                          max_epochs=self._args.max_epochs,
                          min_epochs=self._args.min_epochs,
                          check_val_every_n_epoch=self._args.check_val_every_n_epoch,
                          gradient_clip_val=self._args.gradient_clip_val,
                          val_check_interval=self._args.val_check_interval,
                          enable_progress_bar=self._args.enable_progress_bar,
                          accumulate_grad_batches=self._args.accumulate_grad_batches,)

        self._trainer = trainer

    def _fit(self) -> None:
        self._lit_model.num_data = len(self._data_module.train_data)
        self._trainer.fit(self._lit_model, self._data_module)

    def _select_best_model_from_ckpts(self) -> str:

        # try to get best checkpoint path from ModelCheckpoint callback
        self._logger.info("select model from checkpoints...")
        for callback in self._trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                best_model_path = callback.best_model_path
                if best_model_path and os.path.exists(best_model_path):
                    shutil.copy(best_model_path, self._args.best_ckpt_path)
                    self._logger.info("best model path %s, copied to %s",
                                      best_model_path,
                                      self._args.best_ckpt_path)
                    return self._args.best_ckpt_path

        # if no best checkpoint found (e.g. no dev data), use the last checkpoint
        if self._args.last_ckpt_path and os.path.exists(self._args.last_ckpt_path):
            shutil.copy(self._args.last_ckpt_path, self._args.best_ckpt_path)
            self._logger.warning("no ModelCheckpoint available, use last checkpoint")
            return self._args.best_ckpt_path

        self._logger.error("Unexpected error: No checkpoint available")
        return self._args.best_ckpt_path

    def _get_inf_model(self, checkpoint_path: str):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            self._logger.info("checkpoint [%s] is not valid. use final checkpoint.",
                              checkpoint_path)
        else:
            self._logger.info("generating inference model from %s...", checkpoint_path)
            self._lit_model = BagualuIELitModel.load_from_checkpoint(checkpoint_path,
                                                                     args=self._args,
                                                                     logger=None)
        self._lit_model.eval()

    def _generate_prediction_file(self):

        # generate predictions from dev data
        # if os.path.exists(self._args.dev_data_path):
        #     self._logger.info("predicting on dev data..")
        #     data = list(load_json_list(self._args.dev_data_path))
        #     pred_data = self._generate_prediction_results(data)
        #     dump_json_list(pred_data, os.path.join(self._args.prediction_save_dir,
        #                                            "dev_prediction_results.json"))
        #     eval_results = self._generate_evaluation_results(data, pred_data)
        #     dump_json(eval_results,
        #               os.path.join(self._args.prediction_save_dir,
        #                            "dev_evaluation_results.json"),
        #               indent=4)

        # generate predictions from test data
        if os.path.exists(self._args.test_data_path):
            self._logger.info("predicting on test data..")
            data = list(load_json_list(self._args.test_data_path))
            pred_data = self._generate_prediction_results(data)
            dump_json_list(pred_data, os.path.join(self._args.prediction_save_dir,
                                                   "test_prediction_results.json"))
            eval_results = self._generate_evaluation_results(data, pred_data)
            dump_json(eval_results,
                      os.path.join(self._args.prediction_save_dir,
                                   "test_evaluation_results.json"),
                      indent=4)

    def _generate_prediction_results(self, data: List[dict]) -> List[dict]:
        check_data(data)
        data = data_segment(data)

        batch_size = self._args.batch_size

        result = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i: i + batch_size]
            batch_result = self._extract_model.extract(batch_data, self._lit_model._model) # pylint: disable=protected-access
            result.extend(batch_result)

        result = data_segment_restore(result)

        return result

    def _generate_evaluation_results(self, true_data: List[dict], pred_data: List[dict]):
        eval_results = dict()

        ner_true_data = [d for d in true_data if d["task"] in {"实体识别"}]
        ner_pred_data = [d for d in pred_data if d["task"] in {"实体识别"}]
        if ner_true_data:
            f1, recall, precise = eval_entity_f1(ner_true_data, ner_pred_data)
            eval_results["entity"] = {
                "num_ner_samples": len(ner_true_data),
                "num_ground_truth_entities": recall[1],
                "num_predicted_enttities": precise[1],
                "num_correctly_predicted_entities": precise[0],
                "precision": precise[-1],
                "recall": recall[-1],
                "f1": f1
            }

        re_true_data = [d for d in true_data if d["task"] in {"关系抽取"}]
        re_pred_data = [d for d in pred_data if d["task"] in {"关系抽取"}]
        if re_true_data:
            f1, recall, precise = eval_entity_relation_f1(re_true_data, re_pred_data)
            eval_results["relation"] = {
                "num_re_samples": len(re_true_data),
                "num_ground_truth_relations": recall[1],
                "num_predicted_relations": precise[1],
                "num_correctly_predicted_relations": precise[0],
                "precision": precise[-1],
                "recall": recall[-1],
                "f1": f1
            }

        return eval_results

    def _export_onnx(self) -> None:

        # save tokenizer
        self._tokenizer.save_pretrained(self._args.model_save_dir)
        self._logger.info("tokenizer saved to %s", self._args.model_save_dir)

        # load model config
        model_config = BagualuIEOnnxConfig()
        bert_model = self._lit_model._model.cpu() # pylint: disable=protected-access
        bert_model.eval()

        # get dummy model input
        input_sample = next(iter(self._data_module.train_dataloader()))
        model_input = [input_sample[name] for name in model_config.input_names]

        torch.onnx.export(
            bert_model,
            tuple(model_input),
            f=self._args.onnx_saved_path,
            input_names=model_config.input_names,
            output_names=model_config.output_names,
            dynamic_axes=model_config.dynamic_axes,
            do_constant_folding=True,
            opset_version=13
        )
        self._logger.info("onnx model saved to %s", self._args.onnx_saved_path)
