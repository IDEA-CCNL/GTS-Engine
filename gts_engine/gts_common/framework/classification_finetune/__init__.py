"""句子分类任务基类与特殊工具集合.

包含:
    * BaseTrainingPipelineClf: 句子分类训练Pipeline基类
    * BaseTrainingArgumentsClf: 句子分类训练Pipeline对应参数集合基类
    * BaseInferenceManagerClf: 句子分类推理器基类
    * BaseInferenceArgumentsClf: 句子分类推理器对应参数集合基类
    * BaseDataModuleClf: 句子分类pl_lightning.DataModule基类，主要包含数据选择和加载逻辑
    * BaseDatasetClf: 句子分类torch.Dataset基类，主要包含数据编码逻辑
    * BaseTrainingLightningClf: 句子分类用于训练模型的pl_lightning.LightningModule基类
    * BaseInferenceLightningClf: 句子分类用于推理模型的pl_lightning.LightningModule基类
    * StdLabel: 标签处理器
    * DataReaderClf: 句子分类任务数据读取器
    * consts: 句子分类任务特定常量集合
    * mask_tools: 句子分类任务特定mask工具集

Todo:
    - [ ] (Jiang Yuzhen) 对StdPrompt进行拆分和功能修改，补全docstring
"""
from .base_arguments_clf import (BaseInferenceArgumentsClf,
                                 BaseTrainingArgumentsClf)
from .base_data_module_clf import BaseDataModuleClf
from .base_dataset_clf import BaseDatasetClf
from .base_inference_manager_clf import BaseInferenceManagerClf
from .base_lightnings_clf import (BaseInferenceLightningClf,
                                  BaseTrainingLightningClf)
from .base_training_pipeline_clf import BaseTrainingPipelineClf
from .data_reader_clf import DataReaderClf
from .label import StdLabel

__all__ = [
    "BaseTrainingPipelineClf", "BaseTrainingArgumentsClf",
    "BaseInferenceArgumentsClf", "BaseDataModuleClf",
    "BaseTrainingLightningClf", "BaseInferenceLightningClf", "BaseDatasetClf",
    "BaseInferenceManagerClf", "StdLabel", "DataReaderClf"
]
