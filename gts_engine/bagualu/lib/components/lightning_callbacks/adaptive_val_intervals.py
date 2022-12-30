from typing import Dict, Optional, List
from enum import Enum
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.trainer.states import TrainerFn
from torch import Tensor
from math import ceil
from pytorch_lightning.callbacks import Callback
from abc import ABCMeta, abstractmethod

from ...utils.statistics import ExponentialSmoothingList, DynamicMax, interval_mean
from ...framework.mixin import OptionalLoggerMixin

#############################################################################################
## Base
#############################################################################################

class BaseAdaptiveValidationInterval(Callback, OptionalLoggerMixin, metaclass=ABCMeta):
    
    def __init__(self, logger_name: Optional[str] = None):
        Callback.__init__(self)
        OptionalLoggerMixin.__init__(self, logger_name)
        
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if not trainer.state.fn == TrainerFn.FITTING:
            return
        self._disable_validation(trainer)  
    
    def on_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.state.fn == TrainerFn.FITTING:
            return
        if self._get_validation_flag(trainer, pl_module):
            self.info("trigger validation")
            self._enable_validation(trainer)
    
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.state.fn == TrainerFn.FITTING:
            return
        self._disable_validation(trainer)
    
    def _disable_validation(self, trainer: "pl.Trainer"):
        """使loop不进行验证"""
        trainer.fit_loop.epoch_loop._should_check_val_fx = lambda : False   
    
    def _enable_validation(self, trainer: "pl.Trainer"):
        """使loop进行验证"""
        trainer.fit_loop.epoch_loop._should_check_val_fx = lambda : True 
        
    @abstractmethod
    def _get_validation_flag(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> bool: ...

#############################################################################################
## Subs
#############################################################################################

class ADAPTIVE_VAL_INTERVAL_MODE(Enum):
    ADAPTIVE = 1
    FIXED = 2


class AdaptiveValIntervalFixed(BaseAdaptiveValidationInterval):
    
    def __init__(self, train_sample_num: int, batch_size: int, logger_name: Optional[str] = None):
        super().__init__(logger_name)
        batch_num_per_epoch = train_sample_num // batch_size
        val_times_per_epoch = 7
        self.__val_interval = max(1, batch_num_per_epoch // val_times_per_epoch)
    
    def _get_validation_flag(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> bool:
        return trainer.global_step % self.__val_interval == 0
    
    
class AdaptiveValIntervalDevLoss(BaseAdaptiveValidationInterval):
    
    def __init__(self, train_sample_num: int, batch_size: int, logger_name: Optional[str] = None):
        super().__init__(logger_name)
        self.__batch_size = batch_size
        self.__train_sample_num = train_sample_num
        self.__current_dev_times_per_epoch = 1
        self.__min_dev_times_per_epoch = 3
        self.__max_dev_times_per_epoch = 7
        self.__batch_num_each_epoch = self.__train_sample_num // self.__batch_size
        self.__val_interval = max(int(self.__batch_num_each_epoch // self.__current_dev_times_per_epoch), 1)
        self.__next_validation = self.__val_interval
        self.__dev_loss_list: List[float] = []
        
    def _get_validation_flag(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> bool:
        return trainer.global_step == self.__next_validation
    
    def __step(self, loss: float) -> None:
        self.__dev_loss_list.append(loss)
        if self.__loss_rising_flag():
            self.__current_dev_times_per_epoch -= 2
            self.__current_dev_times_per_epoch = max(self.__current_dev_times_per_epoch, self.__min_dev_times_per_epoch)
        else:
            self.__current_dev_times_per_epoch += 2
            self.__current_dev_times_per_epoch = min(self.__current_dev_times_per_epoch, self.__max_dev_times_per_epoch)
        self.__val_interval = max(int(self.__batch_num_each_epoch // self.__current_dev_times_per_epoch), 5)
    
    def __loss_rising_flag(self, window=2, pre_sample_len=4):
        """
        判断当前loss是否处于上涨状态
        """
        dev_sample_start = len(self.__dev_loss_list) - window if (len(self.__dev_loss_list) - window) >= 0 else 0
        pre_sample_start = len(self.__dev_loss_list) - window - pre_sample_len if (len(self.__dev_loss_list) - window - pre_sample_len) >= 0 else 0
        pre_sample_end = pre_sample_start + window if pre_sample_start + window <= len(self.__dev_loss_list) else len(self.__dev_loss_list)
        pre_step_loss_avg = np.mean(self.__dev_loss_list[pre_sample_start:pre_sample_end])
        dev_step_loss_avg = np.mean(self.__dev_loss_list[dev_sample_start:])
        return True if pre_step_loss_avg - dev_step_loss_avg < 0.00 else False
    
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.state.fn != TrainerFn.FITTING:
            return
        super().on_validation_epoch_end(trainer, pl_module)
        dev_loss = float(trainer.logged_metrics["dev_loss"].item())
        self.__step(dev_loss)
        self.__next_validation = trainer.global_step + self.__val_interval
        self.info(f"current validation interval: {self.__val_interval}, next validation will be at step {self.__next_validation}")


class AdaptiveValIntervalTrainLoss(BaseAdaptiveValidationInterval):
    
    def __init__(self, train_sample_num: int, batch_size: int, logger_name: Optional[str] = None):
        super().__init__(logger_name)
        self.__dev_gate: bool = False
        self.__batch_num_per_epoch = train_sample_num // batch_size
        val_times_per_epoch = 7
        self.__val_interval = max(self.__batch_num_per_epoch // val_times_per_epoch, 1)
        self.__train_loss_smooth_list = ExponentialSmoothingList(level=2, alpha=0.9, warmup_steps=10)
        self.__diff_smooth_list = ExponentialSmoothingList(level=2, alpha=0.9, warmup_steps=10)
        self.__diff_ratio_max = DynamicMax(top_n=3)
        self.__sample_gap = max(min(15 // max(int((batch_size // 8) ** 0.5), 1), self.__batch_num_per_epoch // 8), 1)
        self.__trigger_count = 0 # 连续触发次数
        self.__trigger_count_thres = 2 # 连续触发次数阈值
        self.__ratio_thres = 0.8 # 比例阈值 # TODO: 根据batch_size放缩，需要更多实验支撑
        self.__warmup_steps = self.__batch_num_per_epoch // 2
                
    def _get_validation_flag(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> bool:
        return self.__dev_gate and trainer.global_step % self.__val_interval == 0
    
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Dict[str, Tensor], batch, batch_idx: int) -> None:
        if trainer.state.fn != TrainerFn.FITTING:
            return
        if self.__dev_gate:
            return
        train_loss: float = outputs["loss"].item()
        if trainer.global_step % self.__sample_gap == 0:
            self.__step(train_loss, trainer, pl_module)
        
    def __step(self, train_loss: float, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch >= trainer.max_epochs - 3:
            self.info(f"start to validate every {self.__val_interval} steps")
            self.__dev_gate = True
            return
        self.__train_loss_smooth_list.append(train_loss)
        if len(self.__train_loss_smooth_list) <= 2:
            return
        diff = self.__train_loss_smooth_list[2][-1] - self.__train_loss_smooth_list[2][-2]
        self.__diff_smooth_list.append(diff)
        if trainer.global_step > self.__warmup_steps:
            if self.__trigger(trainer, pl_module):
                self.__trigger_count += 1
            else:
                self.__trigger_count = 0
            if self.__trigger_count >= self.__trigger_count_thres:
                self.__dev_gate = True
                self.info(f"start to validate every {self.__val_interval} steps")
    
    def __trigger(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        diff_ratio = self.__diff_smooth_list[1][-1] / self.__warm_up_mean_diff
        self.__diff_ratio_max.step(diff_ratio)
        normalized_diff_ratio = diff_ratio / self.__diff_ratio_max.max
        pl_module.log(f"diff_ratio", diff_ratio)
        pl_module.log(f"normalized_diff_ratio", normalized_diff_ratio)
        return normalized_diff_ratio < self.__ratio_thres
    
    __warm_up_mean_diff_: Optional[float] = None
    @property
    def __warm_up_mean_diff(self) -> float:
        if self.__warm_up_mean_diff_ is None:
            self.__warm_up_mean_diff_ = interval_mean(self.__diff_smooth_list[1][:self.__warmup_steps])
        return self.__warm_up_mean_diff_