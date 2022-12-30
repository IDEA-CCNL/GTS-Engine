from abc import abstractmethod, ABCMeta
from pytorch_lightning import LightningDataModule
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import List, Optional, Literal, Sequence
from torch.utils.data import DataLoader
import os
import math
from collections import Counter
from sklearn.model_selection import train_test_split
import random

from ...components import LoggerManager
from .prompt import StdPrompt
from .consts import Label2Token, LabeledSample, UnlabeledSample, InfSampleProto
from .base_arguments_clf import BaseTrainingArgumentsClf
from ..consts import TRAINING_STAGE
from .base_dataset_clf import BaseDatasetClf
from .data_reader_clf import DataReaderClf

    
class BaseDataModuleClf(LightningDataModule, metaclass=ABCMeta):
    
    def __init__(
        self,
        args: BaseTrainingArgumentsClf,
        prompt: StdPrompt,
        tokenizer: PreTrainedTokenizer
    ):
        super().__init__()
        self._args = args
        self._prompt = prompt
        self._tokenizer = tokenizer
        self._logger = LoggerManager.get_logger(self._args.logger)
        
    def train_dataloader(self, load_ratio: float = 1):
        sample_list = self.train_sample_list[:int(len(self.train_sample_list) * load_ratio)]
        self._logger.info(f"number of training sample: {len(sample_list)}")
        lg_dataset = self._get_train_dataset(sample_list)
        return DataLoader(dataset=lg_dataset, batch_size=self._args.train_batch_size, num_workers=self._args.num_workers * self._args.device_num, shuffle=True)
        
    def val_dataloader(self, stage: Literal[TRAINING_STAGE.VALIDATION, TRAINING_STAGE.INFERENCE], load_ratio: float = 1, resample_thres: Optional[int] = None):
        if self.dev_sample_num == 0:
            return None
        sample_list = self.dev_sample_list[:int(len(self.dev_sample_list) * load_ratio)]
        if resample_thres:
            sample_list = self._resample_sample_list(sample_list, resample_thres)
        if stage == TRAINING_STAGE.VALIDATION:
            lg_dataset = self._get_test_dataset(sample_list)
            return DataLoader(dataset=lg_dataset, batch_size=self._args.valid_batch_size, num_workers=self._args.num_workers * self._args.device_num)
        elif stage == TRAINING_STAGE.INFERENCE:
            lg_dataset = self._get_inf_dataset(sample_list)
            return DataLoader(dataset=lg_dataset, batch_size=self._args.valid_batchsize_per_device, num_workers=self._args.num_workers)
        else:
            raise Exception("stage is not in [TRAINING_STAGE.VALIDATION, TRAINING_STAGE.INFERENCE]")
    
    def test_dataloader(self, stage: Literal[TRAINING_STAGE.TEST, TRAINING_STAGE.INFERENCE], load_ratio: float = 1):
        sample_list = self.test_sample_list[:int(len(self.test_sample_list) * load_ratio)]
        if stage == TRAINING_STAGE.TEST:
            lg_dataset = self._get_test_dataset(sample_list)
            return DataLoader(dataset=lg_dataset, batch_size=self._args.test_batch_size, num_workers=self._args.num_workers * self._args.device_num)
        elif stage == TRAINING_STAGE.INFERENCE:
            lg_dataset = self._get_inf_dataset(sample_list)
            return DataLoader(dataset=lg_dataset, batch_size=self._args.test_batchsize_per_device, num_workers=self._args.num_workers)
        else:
            raise Exception("stage is not in [TRAINING_STAGE.VALIDATION, TRAINING_STAGE.INFERENCE]")
    
    def online_test_dataloader(self, load_ratio: float = 1):
        sample_list = self.online_test_sample_list[:int(len(self.online_test_sample_list) * load_ratio)]
        lg_dataset = self._get_inf_dataset(sample_list)
        return DataLoader(dataset=lg_dataset, batch_size=self._args.test_batchsize_per_device, num_workers=self._args.num_workers)
            
    @property
    def train_sample_num(self) -> int:
        return len(self.train_sample_list)
    
    @property
    def dev_sample_num(self) -> int:
        return len(self.dev_sample_list)
    
    @property
    def class_num(self) -> int:
        return len(self._prompt.label2token)
    
    _train_sample_list: Optional[List[LabeledSample]] = None
    @property
    def train_sample_list(self) -> List[LabeledSample]:
        if self._train_sample_list is None:
            self._train_sample_list = self._load_train_sample_list()
        return self._train_sample_list
    
    _dev_sample_list: Optional[List[LabeledSample]] = None
    @property
    def dev_sample_list(self) -> List[LabeledSample]:
        if self._dev_sample_list is None:
            self._dev_sample_list = self._load_dev_sample_list()
        return self._dev_sample_list
    
    _test_sample_list: Optional[List[LabeledSample]] = None
    @property
    def test_sample_list(self) -> List[LabeledSample]:
        """离线测试数据集"""
        if self._test_sample_list is None:
            self._test_sample_list = self._load_test_sample_list()
        return self._test_sample_list
    
    _online_test_sample_list: Optional[List[UnlabeledSample]] = None
    @property
    def online_test_sample_list(self) -> List[UnlabeledSample]:
        """推理数据集"""
        if self._online_test_sample_list is None:
            self._online_test_sample_list = self._load_online_test_sample_list()
        return self._online_test_sample_list
    
    def _resample_sample_list(self, sample_list: List[LabeledSample], resample_thres: int = 1000) -> List[LabeledSample]:
        """验证数据过多时，进行重新抽样"""
        sample_num = len(sample_list)
        len_t = min(sample_num, resample_thres)

        split_rate = min(len_t / sample_num, 1.0)
        if split_rate == 1.0:
            return sample_list

        labels = [sample.label_id_clf for sample in sample_list]
        label_lt_five = [
            ele[0] for ele in Counter(labels).items() if ele[1] <= 5
        ]
        sample_lt_five = [
            sample for sample in sample_list
            if sample.label_id_clf in label_lt_five
        ]
        sample_more_five = [
            sample for sample in sample_list
            if sample.label_id_clf not in label_lt_five
        ]
        labels_more_five = [
            sample.label_id_clf for sample in sample_more_five
        ]
        _, new_dev, _, _ = train_test_split(sample_more_five,
                                            labels_more_five,
                                            test_size=split_rate,
                                            stratify=labels_more_five)
        new_dev.extend(sample_lt_five)
        self._logger.info('Size of new dev dataset: {}'.format(len(new_dev)))

        return new_dev
    
    #############################################################################################
    ## abstract
    #############################################################################################
    
    @abstractmethod
    def _get_train_dataset(self, sample_list: Sequence[LabeledSample]) -> BaseDatasetClf:
        """根据需要加载训练Dataset"""
        
    @abstractmethod
    def _get_test_dataset(self, sample_list: Sequence[LabeledSample]) -> BaseDatasetClf:
        """根据需要加载测试Dataset"""
        
    @abstractmethod
    def _get_inf_dataset(self, sample_list: Sequence[InfSampleProto]) -> BaseDatasetClf:
        """根据需要加载推理Dataset"""
    
    def _load_train_sample_list(self) -> List[LabeledSample]:
        """加载训练Sample列表"""
        train_data_path = self._args.train_data_path
        if train_data_path is not None and os.path.exists(train_data_path):
            return list(DataReaderClf.read_labeled_sample(self._args.train_data_path, self._prompt.label2token))
        else:
            raise Exception("no training data is passed")
        
    def _load_dev_sample_list(self) -> List[LabeledSample]:
        """加载验证Sample列表"""
        dev_data_path = self._args.dev_data_path
        if dev_data_path is not None and os.path.exists(dev_data_path):
            return list(DataReaderClf.read_labeled_sample(dev_data_path, self._prompt.label2token))
        else:
            return []
    
    def _load_test_sample_list(self) -> List[LabeledSample]:
        """加载离线测试Sample列表"""
        test_data_path = self._args.test_data_path
        if test_data_path is not None and os.path.exists(test_data_path):
            return list(DataReaderClf.read_labeled_sample(test_data_path, self._prompt.label2token))
        else:
            raise Exception("no valid test data path is passed")
    
    def _load_online_test_sample_list(self) -> List[UnlabeledSample]:
        """加载在线推理Sample列表"""
        online_test_data_path = self._args.online_test_data_path
        if online_test_data_path is not None and os.path.exists(online_test_data_path):
            return list(DataReaderClf.read_unlabeled_sample(online_test_data_path))
        else:
            raise Exception("no valid online test data path is passed")
        
    def reparse_args(self):
        """根据数据信息更改训练参数"""
        if len(self._prompt.label_ids) <= 5:
            self._args.label_guided_rate = 0.3
        step_per_epoch = self.train_sample_num // self._args.train_batch_size
        steps = step_per_epoch * self._args.epoch
        min_steps = min(1400 // self._args.device_num, 40 * step_per_epoch)
        if steps < min_steps:
            self._args.epoch = math.ceil(min_steps // step_per_epoch)
        if self.dev_sample_num == 0:
            self._args.epoch = int(0.75 * self._args.epoch)
        self._logger.info(f"reparsing epoch: {self._args.epoch}")
        self._logger.info(f"batch size per device: {self._args.train_batchsize_per_device}")
        self._logger.info(f"total batch size: {self._args.train_batch_size}")