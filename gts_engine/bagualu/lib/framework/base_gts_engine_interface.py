from abc import abstractmethod, ABCMeta, abstractproperty
from typing import Type, List, Optional, Dict, Literal
from pydantic import BaseModel, DirectoryPath, FilePath
from pathlib import Path
from argparse import Namespace
from enum import Enum
import os

from .base_training_pipeline import BaseTrainingPipeline
from .base_inference_manager import BaseInferenceManager

PathStr = str

class GtsEngineArgs(Namespace):
    engine_type: Literal["qiankunding", "bagualu"]
    train_mode: Literal["fast", "standard", "advanced"]
    task_dir: PathStr
    task_type: Literal["classification", "similarity", "nli"]
    num_workers: int
    train_batchsize: int
    valid_batchsize: int
    test_batchsize: int
    max_len: int
    pretrained_model_dir: PathStr
    data_dir: PathStr
    train_data: str
    valid_data: str
    test_data: str
    label_data: str
    save_path: PathStr
    seed: int
    lr: float
    gpus: int
    num_sanity_val_steps: int
    accumulate_grad_batches: int
    val_check_interval: float
    
    @property
    def train_data_path(self) -> Path:
        return Path(self.data_dir) / self.train_data
    
    @property
    def valid_data_path(self) -> Path:
        return Path(self.data_dir) / self.valid_data
    
    @property
    def test_data_path(self) -> Optional[Path]:
        return Path(self.data_dir) / self.test_data if hasattr(self, "test_data") else None
    
    @property
    def label_data_path(self) -> Optional[Path]:
        return Path(self.data_dir) / self.label_data if hasattr(self, "label_data") else None
    
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
    
class ENGINE_TYPE(Enum):
    QKD = "qiankunding"
    BGL = "bagualu"

class BaseGtsEngineInterface(metaclass=ABCMeta):
    """适配GTS-Engine的胶水模块基类，根据GTS-Engine的参数生成对应的bagualu模块"""
    
    def generate_training_pipeline(self, args: GtsEngineArgs) -> BaseTrainingPipeline: 
        """通过GTS-Engine参数实例化bagualu TrainingPipeline"""
        parsed_args_list = self._parse_training_args(args)
        print(f"\n------------------------- parsed args for {self._training_pipeline_type.__name__} -------------------------")
        print(f"\n{' '.join(parsed_args_list)}\n")
        print("------------------------------------------------------------------------------------------\n")
        return self._training_pipeline_type(parsed_args_list)
    
    def prepare_training(self, args: GtsEngineArgs) -> None:
        """训练准备处理，如数据格式转换等"""
        return None
    
    def generate_inference_manager(self, args: GtsEngineArgs) -> BaseInferenceManager:
        """通过GTS-Engine参数实例化agualu InferenceManager"""
        parsed_args_list = self._parse_inference_args(args)
        print(f"\n------------------------- parsed args for {self._inference_manager_type.__name__} -------------------------")
        print(f"\n{' '.join(parsed_args_list)}\n")
        print("------------------------------------------------------------------------------------------\n")
        inference_manager = self._inference_manager_type(parsed_args_list)
        inference_manager.prepare_inference()
        return inference_manager
    
    def prepare_inference(self, args: GtsEngineArgs) -> None:
        """推理准备处理，如数据格式转换等"""
        return None
    
    ########################### abstract ################################
    
    @abstractproperty
    def _training_pipeline_type(self) -> Type[BaseTrainingPipeline]:
        """对应bagualu TrainingPipeline类"""
        ...
    
    @abstractmethod
    def _parse_training_args(self, args: GtsEngineArgs) -> List[str]:
        """将GTS-Engine参数解析为bagualu TrainingPipeline启动参数字符串列表"""
        
    @abstractproperty
    def _inference_manager_type(self) -> Type[BaseInferenceManager]:
        """对应bagualu InferenceManager 类"""
        ...
        
    @abstractmethod
    def _parse_inference_args(self, args: GtsEngineArgs) -> List[str]:
        """将GTS-Engine参数解析为bagualu InferenceManager启动参数字符串列表"""