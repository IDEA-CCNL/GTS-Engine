"""将bagualu嵌入gts-engine的胶水层模块.

通过将gts-engine的参数解析为bagualu模块的参数的中间层，以达到不修改bagualu内部代码将
bagualu嵌入到gts-engine中的目的。

Todo:
    - [ ] (Jiang Yuzhen) 将lib作为更高一级的公共库、并且bagualu不再需要作为独立项目
        来维护之后，可以直接用统一的接口启动qiankunding和bagualu，而不是用胶水层的方式。
"""
from abc import ABCMeta, abstractmethod, abstractproperty
from enum import Enum
from typing import List, Type

from gts_common.arguments import GtsEngineArgs

from .base_inference_manager import BaseInferenceManager
from .base_training_pipeline import BaseTrainingPipeline


class TRAIN_MODE(Enum):
    DEFAULT = "default"
    STD = "standard"
    FAST = "fast"
    ADV = "advanced"


class TASK(Enum):
    CLS = "classification"
    NLI = "nli"
    SIM = "similarity"
    IE = "ie"
    SMY = "summary"


class ENGINE_TYPE(Enum):
    QKD = "qiankunding"
    BGL = "bagualu"


class BaseGtsEngineInterface(metaclass=ABCMeta):
    """适配GTS-Engine的胶水模块基类，根据GTS-Engine的参数生成对应的bagualu模块."""

    def generate_training_pipeline(
            self, args: GtsEngineArgs) -> BaseTrainingPipeline:
        """通过GTS-Engine参数实例化bagualu TrainingPipeline."""
        parsed_args_list = self._parse_training_args(args)
        print(f"\n------------------------- parsed args for "
              f"{self._training_pipeline_type.__name__} ----"
              f"---------------------")
        print(f"\n{' '.join(parsed_args_list)}\n")
        print("----------------------------------------------"
              "--------------------------------------------\n")
        return self._training_pipeline_type(parsed_args_list)

    def prepare_training(self, args: GtsEngineArgs) -> None:
        """训练准备处理，如数据格式转换等."""
        return None

    def generate_inference_manager(
            self, args: GtsEngineArgs) -> BaseInferenceManager:
        """通过GTS-Engine参数实例化agualu InferenceManager."""
        parsed_args_list = self._parse_inference_args(args)
        print(f"\n------------------------- parsed args for "
              f"{self._inference_manager_type.__name__} ----"
              f"---------------------")
        print(f"\n{' '.join(parsed_args_list)}\n")
        print("----------------------------------------------"
              "--------------------------------------------\n")
        inference_manager = self._inference_manager_type(parsed_args_list)
        inference_manager.prepare_inference()
        return inference_manager

    def prepare_inference(self, args: GtsEngineArgs) -> None:
        """推理准备处理，如数据格式转换等."""
        return None

    # ========================== abstract ===============================

    @abstractproperty
    def _training_pipeline_type(self) -> Type[BaseTrainingPipeline]:
        """对应bagualu TrainingPipeline类."""
        ...

    @abstractmethod
    def _parse_training_args(self, args: GtsEngineArgs) -> List[str]:
        """将GTS-Engine参数解析为bagualu TrainingPipeline启动参数字符串列表."""

    @abstractproperty
    def _inference_manager_type(self) -> Type[BaseInferenceManager]:
        """对应bagualu InferenceManager 类."""
        ...

    @abstractmethod
    def _parse_inference_args(self, args: GtsEngineArgs) -> List[str]:
        """将GTS-Engine参数解析为bagualu InferenceManager启动参数字符串列表."""
