from abc import abstractmethod, ABCMeta, abstractproperty
from typing import Type, List

from bagualu.gts_student_lib.framework import BaseTrainingPipeline, BaseInferenceEngine
from bagualu.gts_student_ft_std import FtStdTrainingPipeline
from consts import GTSEngineTrainArgs, GTSEngineInfArgs

#############################################################################################
## Base
#############################################################################################

class BaseBGLModuleFatrory(metaclass=ABCMeta):
    
    def generate_training_pipeline(self, args: GTSEngineTrainArgs) -> BaseTrainingPipeline: 
        """通过GTS-Engine训练参数实例化bagualu TrainingPipeline"""
        return self._training_pipeline_cls(self._parse_training_args(args))
    
    def generate_inference_engine(self, args: GTSEngineInfArgs) -> BaseInferenceEngine:
        """通过GTS-Engine推理参数实例化agualu InferenceEngine"""
        return self._inference_engine_cls(self._parse_inference_args(args))
    
    @abstractproperty
    def _training_pipeline_cls(self) -> Type[BaseTrainingPipeline]:
        """对应bagualu TrainingPipeline类"""
        ...
    
    @abstractmethod
    def _parse_training_args(self, args: GTSEngineTrainArgs) -> List[str]:
        """将GTS-Engine训练参数解析为bagualu TrainingPipeline启动参数字符串列表"""
        
    @abstractproperty
    def _inference_engine_cls(self) -> Type[BaseInferenceEngine]:
        """对应bagualu InferenceEngine 类"""
        ...
        
    @abstractmethod
    def _parse_inference_args(self, args: GTSEngineInfArgs) -> List[str]:
        """将GTS-Engine推理参数解析为bagualu InferenceEngine启动参数字符串列表"""
        
#############################################################################################
## Derived
#############################################################################################

class CLS_STD_ModuleFactory(BaseBGLModuleFatrory):
    
    @property
    def _training_pipeline_cls(self):
        return FtStdTrainingPipeline
    
    @property
    def _inference_engine_cls(self):
        ...
    
    def _parse_training_args(self, args: GTSEngineTrainArgs) -> List[str]:
        return super()._parse_training_args(args)
    
    def _parse_inference_args(self, args: GTSEngineInfArgs) -> List[str]:
        ...
        
    
    
    
    
