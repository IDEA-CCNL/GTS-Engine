from typing import List, Literal, Tuple, Type, Callable, Dict
from argparse import Namespace
from enum import Enum

from gts_common.registry import PIPELINE_REGISTRY
from gts_common.consts import GTSEngineTrainArgs, TRAIN_MODE
from gts_common.bagualu_module_factory import CLS_STD_ModuleFactory, BaseBGLModuleFatrory
from bagualu.gts_student_ft_std.training_pipeline import FtStdTrainingPipeline, FtTrainingPipeline


mode_to_module_factory: Dict[TRAIN_MODE, BaseBGLModuleFatrory] = {
    TRAIN_MODE.STD: CLS_STD_ModuleFactory()
}

@PIPELINE_REGISTRY.register(suffix=__name__) # type: ignore
def train_pipeline(args: GTSEngineTrainArgs):
    train_mode = TRAIN_MODE(args.train_mode)
    module_factory = mode_to_module_factory[train_mode]
    module_factory.generate_training_pipeline(args).main()
    
# @PIPELINE_REGISTRY.register(suffix=__name__)
# def prepare_inference(save_path):
#     train_mode = TRAIN_MODE(args.train_mode)
#     module_factory = mode_to_module_factory[train_mode]