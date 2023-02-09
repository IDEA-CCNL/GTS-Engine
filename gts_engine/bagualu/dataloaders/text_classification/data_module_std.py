import math
import os
from typing import List

from bagualu.arguments.text_classification.arguments_std import \
    TrainingArgumentsClfStd
from bagualu.dataloaders.text_classification.datasets_std import (
    InfDatasetClfStd, TestDatasetClfStd, TrainDatasetClfStd)
from gts_common.components.batch_sampler import PairBatchSampler
from gts_common.components.eda_utils import EDA
from gts_common.framework.classification_finetune import (BaseDataModuleClf,
                                                          DataReaderClf)
from gts_common.framework.classification_finetune.consts import LabeledSample
from torch.utils.data import DataLoader


class DataModuleClfStd(BaseDataModuleClf):

    _args: TrainingArgumentsClfStd

    def _get_train_dataset(self, sample_list):
        return TrainDatasetClfStd(sample_list, self._tokenizer, self._label,
                                  self._args.prefix_prompt,
                                  self._args.training_prompt,
                                  self._args.label_guided_rate,
                                  self._args.max_length,
                                  self._args.wwm_mask_rate)

    def _get_test_dataset(self, sample_list):
        return TestDatasetClfStd(sample_list, self._tokenizer, self._label,
                                 self._args.inference_prompt,
                                 self._args.prefix_prompt,
                                 self._args.max_length)

    def _get_inf_dataset(self, sample_list):
        return InfDatasetClfStd(sample_list, self._tokenizer, self._label,
                                self._args.inference_prompt,
                                self._args.prefix_prompt,
                                self._args.max_length)

    def _load_train_sample_list(self):
        sample_list: List[LabeledSample] = []
        if self._args.train_data_path is not None:
            sample_list += list(
                DataReaderClf.read_labeled_sample(self._args.train_data_path,
                                                  self._label.label2token))
        else:
            raise Exception("no training data is passed")
        if self._args.aug_eda_gate:
            if self._args.aug_eda_path is None:
                raise Exception("eda augmentation file path is not passed")
            eda: EDA[LabeledSample] = EDA(tokenizer=self._tokenizer,
                                          logger_name=self._args.logger)
            sample_list += eda.eda_aug(sample_list, self._args.aug_eda_path)
        return sample_list

    def _load_raw_train_sample_list(self) -> List[LabeledSample]:
        sample_list: List[LabeledSample] = []
        if self._args.train_data_path is not None:
            sample_list += list(
                DataReaderClf.read_labeled_sample(self._args.train_data_path,
                                                  self._label.label2token))
        else:
            raise Exception("no training data is passed")
        return sample_list

    def raw_train_dataloader(self, load_ratio: float = 1):
        sample_list = self._load_raw_train_sample_list()
        self._logger.info(f"number of training sample: {len(sample_list)}")
        lg_dataset = self._get_train_dataset(sample_list)
        return DataLoader(dataset=lg_dataset,
                          batch_size=self._args.train_batch_size,
                          num_workers=6 * self._args.device_num,
                          shuffle=True)

    def reparse_args(self):
        """根据数据信息更改训练参数."""
        if len(self._label.label_ids) <= 5:
            self._args.label_guided_rate = 0.3
        step_per_epoch = self.train_sample_num // self._args.train_batch_size
        steps = step_per_epoch * self._args.max_epochs
        min_steps = min(1400 // self._args.device_num, 40 * step_per_epoch)
        if steps < min_steps:
            self._args.max_epochs = math.ceil(min_steps // step_per_epoch)
        if self.dev_sample_num == 0:
            self._args.max_epochs = int(0.75 * self._args.max_epochs)
        if self._args.max_epochs < self._args.min_epochs:
            self._args.min_epochs = self._args.max_epochs
        self._logger.info(f"reparsing epoch: {self._args.max_epochs}")
        self._logger.info(
            f"batch size per device: {self._args.train_batchsize_per_device}")
        self._logger.info(f"total batch size: {self._args.train_batch_size}")

        if len(self._label.label_ids) < self._args.rdrop_gate:
            # 类别数<rdrop gate时，使用rdrop
            self._args.use_rdrop = True
        if self._args.dev_data_path is not None and os.path.exists(
                self._args.dev_data_path) and len(
                    self._label.label_ids) > self._args.rdrop_gate:
            # 类别数大于rdrop gate时(knn在小类别任务有负向作用)，且有验证集时，才使用knn
            self._args.use_knn = True

    def train_dataloader(self, load_ratio: float = 1):
        sample_list = self.train_sample_list[:int(
            len(self.train_sample_list) * load_ratio)]
        self._logger.info(f"number of training sample: {len(sample_list)}")
        lg_dataset = self._get_train_dataset(sample_list)
        # cs-kd: 实现pairbatch,实现同类别采样
        sampler = PairBatchSampler(lg_dataset, self._args.train_batch_size)
        if self._args.use_knn:
            return DataLoader(
                dataset=lg_dataset,
                batch_sampler=sampler(lg_dataset),  # type: ignore
                num_workers=6 * self._args.device_num)
        else:
            return DataLoader(dataset=lg_dataset,
                              batch_size=self._args.train_batch_size,
                              num_workers=6 * self._args.device_num,
                              shuffle=True)
