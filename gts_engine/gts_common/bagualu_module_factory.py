from abc import abstractmethod, ABCMeta, abstractproperty
from typing import Type, List, Optional, Dict
from pydantic import BaseModel, DirectoryPath, FilePath
from pathlib import Path
import os

from bagualu.gts_student_lib.framework import BaseTrainingPipeline, BaseInferenceEngine
from bagualu.gts_student_ft_std import FtStdTrainingPipeline, FtStdInferenceEngine
from bagualu.gts_student_lib.utils.json_utils import load_json, dump_json
from .consts import GTSEngineArgs, TRAIN_MODE

#############################################################################################
## Base
#############################################################################################

class BaseBGLModuleFatrory(metaclass=ABCMeta):
    """胶水模块基类，根据GTS-Engine的参数生成对应的bagualu模块"""
    
    def generate_training_pipeline(self, args: GTSEngineArgs) -> BaseTrainingPipeline: 
        """通过GTS-Engine参数实例化bagualu TrainingPipeline"""
        parsed_args_list = self._parse_training_args(args)
        print(f"\n------------------------- parsed args for {self._training_pipeline_cls.__name__} -------------------------")
        print(f"\n{' '.join(parsed_args_list)}\n")
        print("------------------------------------------------------------------------------------------\n")
        return self._training_pipeline_cls(parsed_args_list)
    
    def prepare_training(self, args: GTSEngineArgs) -> None:
        """训练准备处理，如数据格式转换等"""
        return None
    
    def generate_inference_engine(self, args: GTSEngineArgs) -> BaseInferenceEngine:
        """通过GTS-Engine参数实例化agualu InferenceEngine"""
        parsed_args_list = self._parse_inference_args(args)
        print(f"\n------------------------- parsed args for {self._inference_engine_cls.__name__} -------------------------")
        print(f"\n{' '.join(parsed_args_list)}\n")
        print("------------------------------------------------------------------------------------------\n")
        inference_engine = self._inference_engine_cls(parsed_args_list)
        inference_engine.prepare_inference()
        return inference_engine
    
    def prepare_inference(self, args: GTSEngineArgs) -> None:
        """推理准备处理，如数据格式转换等"""
        return None
    
    ########################### abstract ################################
    
    @abstractproperty
    def _training_pipeline_cls(self) -> Type[BaseTrainingPipeline]:
        """对应bagualu TrainingPipeline类"""
        ...
    
    @abstractmethod
    def _parse_training_args(self, args: GTSEngineArgs) -> List[str]:
        """将GTS-Engine参数解析为bagualu TrainingPipeline启动参数字符串列表"""
        
    @abstractproperty
    def _inference_engine_cls(self) -> Type[BaseInferenceEngine]:
        """对应bagualu InferenceEngine 类"""
        ...
        
    @abstractmethod
    def _parse_inference_args(self, args: GTSEngineArgs) -> List[str]:
        """将GTS-Engine参数解析为bagualu InferenceEngine启动参数字符串列表"""
        
#############################################################################################
## Derived
#############################################################################################

########################################################################################
##################################### cls-std #####################################

class TypeCheckedTrainArgs(BaseModel):
    """GTS-Engine相关参数进行runtime类型检查与转换"""
    task_dir: DirectoryPath
    pretrained_model_dir: DirectoryPath
    data_dir: DirectoryPath
    save_path: DirectoryPath
    train_data_path: FilePath
    valid_data_path: FilePath
    test_data_path: Optional[FilePath]
    label_data_path: Optional[FilePath]
    gpus: int
    train_mode: TRAIN_MODE
    seed: int
    
class TypeCheckedInfArgs(BaseModel):
    model_save_dir: DirectoryPath
    label2id_path: FilePath
    

class CLS_STD_ModuleFactory(BaseBGLModuleFatrory):
    
    @property
    def _training_pipeline_cls(self):
        return FtStdTrainingPipeline
    
    def _parse_training_args(self, args: GTSEngineArgs) -> List[str]:
        # args类型检查
        type_checked_args = TypeCheckedTrainArgs(
            task_dir=Path(args.task_dir),
            pretrained_model_dir=Path(args.pretrained_model_dir),
            data_dir=Path(args.data_dir),
            save_path=Path(args.save_path),
            train_data_path=args.train_data_path,
            valid_data_path=args.valid_data_path,
            test_data_path=args.test_data_path,
            label_data_path=self.__get_label2id_path(args), # 将在prepare_training()中通过label_data生成label2id.json
            gpus=args.gpus,
            train_mode=TRAIN_MODE(args.train_mode),
            seed=args.seed
        )
        args_parse_list: List[str] = []
        args_parse_list.extend(["--gts_input_path", str(type_checked_args.task_dir)])
        args_parse_list.extend(["--gts_pretrained_model_path", str(type_checked_args.pretrained_model_dir)])
        args_parse_list.extend(["--gts_output_dir", str(type_checked_args.save_path)])
        args_parse_list.extend(["--gts_train_level", "1"])
        args_parse_list.extend(["--gpu_num", str(type_checked_args.gpus)])
        args_parse_list.extend(["--run_mode", "online"])
        args_parse_list.extend(["--train_data_path", str(type_checked_args.train_data_path)])
        args_parse_list.extend(["--dev_data_path", str(type_checked_args.valid_data_path)])
        if type_checked_args.test_data_path is not None:
            args_parse_list.extend(["--test_data_path", str(type_checked_args.test_data_path)])
        try:
            assert type_checked_args.label_data_path is not None
            args_parse_list.extend(["--label2id_path", str(type_checked_args.label_data_path)])
        except:
            raise Exception("you should pass label_data file in classification task")
        args_parse_list.extend(["--log_dir", str(type_checked_args.task_dir / "logs")])
        return args_parse_list
    
    def prepare_training(self, args: GTSEngineArgs) -> None:
        # 将label_data数据转为label2id格式
        assert args.label_data_path is not None
        label_data: Dict[str, List[str]] = load_json(args.label_data_path) # type: ignore
        label2id = {
            label: {"id": idx, "label_desc_zh": label}
            for idx, label in enumerate(label_data["labels"])
        }
        dump_json(label2id, self.__get_label2id_path(args))
            
    @property
    def _inference_engine_cls(self):
        return FtStdInferenceEngine
    
    def _parse_inference_args(self, args: GTSEngineArgs) -> List[str]:
        type_checked_args = TypeCheckedInfArgs(
            label2id_path=self.__get_label2id_path(args),
            model_save_dir=Path(args.task_dir) / "outputs" / "student_output" / "finetune_output"
        )
        args_parse_list: List[str] = []
        args_parse_list.extend(["--model_save_dir", str(type_checked_args.model_save_dir)])
        args_parse_list.extend(["--label2id_path", str(type_checked_args.label2id_path)])
        return args_parse_list
        
    def __get_label2id_path(self, args: GTSEngineArgs) -> FilePath:
        return Path(args.task_dir) / "label2id.json"
    
    
########################################################################################
##################################### cls-fast #####################################
