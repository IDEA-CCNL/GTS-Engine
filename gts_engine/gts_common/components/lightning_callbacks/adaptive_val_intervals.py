"""用于定义自适应验证区间的pytorch lightning callbacks.

Todo:
    - [ ] (Jiang Yuzhen) 进一步实验、调参或寻找理论依据，甚至考虑使用简单模型来学习验证开始时机
"""
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
from gts_common.framework.mixin import OptionalLoggerMixin
from gts_common.utils.statistics import (DynamicMax, ExponentialSmoothingList,
                                         interval_mean)
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import TrainerFn
from torch import Tensor

# =============================================================================
# base
# =============================================================================


class BaseAdaptiveValidationInterval(Callback,
                                     OptionalLoggerMixin,
                                     metaclass=ABCMeta):
    """自适应验证区间Callback基类.

    实现对每个step是否进行验证进行动态控制，子类只需实现接口_get_validation_flag()，在每个
    step计算出当前step是否需要验证，即可通过基类触发验证。
    """

    def __init__(self, logger_name: Optional[str] = None):
        """实例化自适应验证区间Callback.

        Args:
            logger_name (Optional[str], optional):
                输出的logger全局名称，为None则使用print输出。
        """
        Callback.__init__(self)
        OptionalLoggerMixin.__init__(self, logger_name)

    def setup(self,
              trainer: "pl.Trainer",
              pl_module: "pl.LightningModule",
              stage: Optional[str] = None) -> None:
        # 如果不是`trainer.fit()`，则跳过
        if not trainer.state.fn == TrainerFn.FITTING:
            return
        # 关闭验证，后续手动开启
        self._disable_validation(trainer)

    def on_batch_start(self, trainer: "pl.Trainer",
                       pl_module: "pl.LightningModule") -> None:
        # 如果不是`trainer.fit()`，则跳过
        if not trainer.state.fn == TrainerFn.FITTING:
            return
        # 通过self._get_validation_flag(trainer, pl_module)计算当前step是否需要验证并开启验证
        if self._get_validation_flag(trainer, pl_module):
            self.info("trigger validation")
            self._enable_validation(trainer)

    def on_validation_epoch_end(self, trainer: "pl.Trainer",
                                pl_module: "pl.LightningModule") -> None:
        # 如果不是`trainer.fit()`，则跳过
        if not trainer.state.fn == TrainerFn.FITTING:
            return
        # 验证结束时关闭验证，等待下次手动开启
        self._disable_validation(trainer)

    def _disable_validation(self, trainer: "pl.Trainer"):
        """使loop不进行验证."""
        trainer.fit_loop.epoch_loop._should_check_val_fx = lambda: False

    def _enable_validation(self, trainer: "pl.Trainer"):
        """使loop进行验证."""
        trainer.fit_loop.epoch_loop._should_check_val_fx = lambda: True

    @abstractmethod
    def _get_validation_flag(self, trainer: "pl.Trainer",
                             pl_module: "pl.LightningModule") -> bool:
        """计算当前step是否需要验证.

        Args:
            trainer (pl.Trainer): 当前trainer
            pl_module (pl.LightningModule): 当前的lightning_module

        Returns:
            bool: 当前step是否触发验证
        """


# =============================================================================
# subs
# =============================================================================


class ADAPTIVE_VAL_INTERVAL_MODE(Enum):
    """验证策略模式."""
    ADAPTIVE = 1
    FIXED = 2


class AdaptiveValIntervalFixed(BaseAdaptiveValidationInterval):
    """Fixed模式验证策略."""

    def __init__(self,
                 train_sample_num: int,
                 batch_size: int,
                 logger_name: Optional[str] = None):
        super().__init__(logger_name)
        batch_num_per_epoch = train_sample_num // batch_size
        val_times_per_epoch = 7
        self.__val_interval = max(1,
                                  batch_num_per_epoch // val_times_per_epoch)

    def _get_validation_flag(self, trainer: "pl.Trainer",
                             pl_module: "pl.LightningModule") -> bool:
        return trainer.global_step % self.__val_interval == 0


class AdaptiveValIntervalDevLoss(BaseAdaptiveValidationInterval):
    """基于dev loss的Adaptive验证策略."""

    def __init__(self,
                 train_sample_num: int,
                 batch_size: int,
                 logger_name: Optional[str] = None):
        """实例化.

        Args:
            train_sample_num (int):
                训练样本量。
            batch_size (int):
                训练batch_size。
            logger_name (Optional[str], optional):
                输出的logger全局名称，为None则使用print输出。
        """
        super().__init__(logger_name)
        self.__batch_size = batch_size
        self.__train_sample_num = train_sample_num
        self.__current_dev_times_per_epoch = 1
        self.__min_dev_times_per_epoch = 3
        self.__max_dev_times_per_epoch = 7
        self.__batch_num_each_epoch = (self.__train_sample_num //
                                       self.__batch_size)
        self.__val_interval = max(
            int(self.__batch_num_each_epoch //
                self.__current_dev_times_per_epoch), 1)
        self.__next_validation = self.__val_interval
        self.__dev_loss_list: List[float] = []

    def _get_validation_flag(self, trainer: "pl.Trainer",
                             pl_module: "pl.LightningModule") -> bool:
        return trainer.global_step == self.__next_validation

    def on_validation_epoch_end(self, trainer: "pl.Trainer",
                                pl_module: "pl.LightningModule") -> None:
        """每个验证周期结束时，记录dev loss."""
        if trainer.state.fn != TrainerFn.FITTING:
            return
        super().on_validation_epoch_end(trainer, pl_module)
        dev_loss = float(trainer.logged_metrics["dev_loss"].item())
        self.__step(dev_loss)
        self.__next_validation = trainer.global_step + self.__val_interval
        self.info(f"current validation interval: {self.__val_interval}, "
                  f"next validation will be at step {self.__next_validation}")

    def __step(self, loss: float) -> None:
        self.__dev_loss_list.append(loss)
        if self.__loss_rising_flag():
            self.__current_dev_times_per_epoch -= 2
            self.__current_dev_times_per_epoch = max(
                self.__current_dev_times_per_epoch,
                self.__min_dev_times_per_epoch)
        else:
            self.__current_dev_times_per_epoch += 2
            self.__current_dev_times_per_epoch = min(
                self.__current_dev_times_per_epoch,
                self.__max_dev_times_per_epoch)
        self.__val_interval = max(
            int(self.__batch_num_each_epoch //
                self.__current_dev_times_per_epoch), 5)

    def __loss_rising_flag(self, window=2, pre_sample_len=4):
        """判断当前loss是否处于上涨状态."""
        dev_sample_start = (len(self.__dev_loss_list) - window if
                            (len(self.__dev_loss_list) - window) >= 0 else 0)
        pre_sample_start = (
            len(self.__dev_loss_list) - window - pre_sample_len if
            (len(self.__dev_loss_list) - window - pre_sample_len) >= 0 else 0)
        pre_sample_end = (pre_sample_start + window if pre_sample_start +
                          window <= len(self.__dev_loss_list) else len(
                              self.__dev_loss_list))
        pre_step_loss_avg = np.mean(
            self.__dev_loss_list[pre_sample_start:pre_sample_end])
        dev_step_loss_avg = np.mean(self.__dev_loss_list[dev_sample_start:])
        return True if pre_step_loss_avg - dev_step_loss_avg < 0.00 else False


class AdaptiveValIntervalTrainLoss(BaseAdaptiveValidationInterval):
    """基于train loss的Adaptive验证策略.

    间隔多步采样train loss，进行指数平滑并差分，计算差分与warm up阶段的平均差分的均值
    的比例，比例低于阈值则触发一次，连续触发多次则开启验证，随后每固定的interval验证一次。
    """

    def __init__(self,
                 train_sample_num: int,
                 batch_size: int,
                 logger_name: Optional[str] = None):
        """实例化.

        Args:
            train_sample_num (int):
                训练样本量。
            batch_size (int):
                训练batch_size。
            logger_name (Optional[str], optional):
                输出的logger全局名称，为None则使用print输出。
        """
        super().__init__(logger_name)
        self.__dev_gate: bool = False  # 是否触发验证标志
        self.__batch_num_per_epoch = train_sample_num // batch_size
        val_times_per_epoch = 7
        self.__val_interval = max(
            self.__batch_num_per_epoch // val_times_per_epoch, 1)
        self.__train_loss_smooth_list = ExponentialSmoothingList(
            level=2, alpha=0.9, warmup_steps=10)  # train loss的指数平均列表
        self.__diff_smooth_list = ExponentialSmoothingList(
            level=2, alpha=0.9, warmup_steps=10)  # train loss差分的指数平均列表
        self.__diff_ratio_max = DynamicMax(top_n=3)  # 记录第三大的train loss差分比例
        # 采样步数，计算比较复杂是为一些极端情况兜底
        self.__sample_gap = max(
            min(15 // max(int((batch_size // 8)**0.5), 1),
                self.__batch_num_per_epoch // 8), 1)
        self.__trigger_count = 0  # 连续触发次数
        self.__trigger_count_thres = 2  # 连续触发次数阈值，超过阈值则打开self.__dev_gate
        # TODO (Jiang Yuzhen) 根据batch_size放缩self.__ratio_thres，需要更多实验支撑
        self.__ratio_thres = 0.8  # train_loss差分与初始warm up阶段差分的比例阈值，低于阈值则触发一次
        self.__warmup_steps = self.__batch_num_per_epoch // 2  # 初始warm up步数
        self.__warm_up_mean_diff_: Optional[float] = None  # warm up平均差分缓存

    def _get_validation_flag(self, trainer: "pl.Trainer",
                             pl_module: "pl.LightningModule") -> bool:
        return (self.__dev_gate
                and trainer.global_step % self.__val_interval == 0)

    def on_train_batch_end(self, trainer: "pl.Trainer",
                           pl_module: "pl.LightningModule",
                           outputs: Dict[str, Tensor], batch,
                           batch_idx: int) -> None:
        """采样train loss."""
        if trainer.state.fn != TrainerFn.FITTING:
            return
        if self.__dev_gate:
            return
        train_loss: float = outputs["loss"].item()
        if trainer.global_step % self.__sample_gap == 0:
            self.__step(train_loss, trainer, pl_module)

    def __step(self, train_loss: float, trainer: "pl.Trainer",
               pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch >= trainer.max_epochs - 3:
            self.info(f"start to validate every {self.__val_interval} steps")
            self.__dev_gate = True
            return
        self.__train_loss_smooth_list.append(train_loss)
        if len(self.__train_loss_smooth_list) <= 2:
            return
        diff = (self.__train_loss_smooth_list[2][-1] -
                self.__train_loss_smooth_list[2][-2])
        self.__diff_smooth_list.append(diff)
        if trainer.global_step > self.__warmup_steps:
            if self.__trigger(trainer, pl_module):
                self.__trigger_count += 1
            else:
                self.__trigger_count = 0
            if self.__trigger_count >= self.__trigger_count_thres:
                self.__dev_gate = True
                self.info(f"start to validate every"
                          f" {self.__val_interval} steps")

    def __trigger(self, trainer: "pl.Trainer",
                  pl_module: "pl.LightningModule"):
        diff_ratio = self.__diff_smooth_list[1][-1] / self.__warm_up_mean_diff
        self.__diff_ratio_max.step(diff_ratio)
        normalized_diff_ratio = diff_ratio / self.__diff_ratio_max.max
        pl_module.log("diff_ratio", diff_ratio)
        pl_module.log("normalized_diff_ratio", normalized_diff_ratio)
        return normalized_diff_ratio < self.__ratio_thres

    @property
    def __warm_up_mean_diff(self) -> float:
        """warm up阶段train loss的平均差分."""
        if self.__warm_up_mean_diff_ is None:
            self.__warm_up_mean_diff_ = interval_mean(
                self.__diff_smooth_list[1][:self.__warmup_steps])
        return self.__warm_up_mean_diff_
