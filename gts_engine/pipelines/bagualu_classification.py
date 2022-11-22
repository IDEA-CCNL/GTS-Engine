from typing import List, Literal, Tuple, Type, Callable, Dict
from argparse import Namespace
from enum import Enum

from gts_common.registry import PIPELINE_REGISTRY
from gts_common.consts import GTSEngineArgs, TRAIN_MODE
from gts_common.bagualu_module_factory import CLS_STD_ModuleFactory, BaseBGLModuleFatrory
from bagualu.gts_student_lib.framework import BaseInferenceEngine
from gts_common.pipeline_utils import load_args


mode_to_module_factory: Dict[TRAIN_MODE, BaseBGLModuleFatrory] = {
    TRAIN_MODE.STD: CLS_STD_ModuleFactory()
}

@PIPELINE_REGISTRY.register(suffix=__name__) # type: ignore
def train_pipeline(args: GTSEngineArgs):
    train_mode = TRAIN_MODE(args.train_mode)
    module_factory = mode_to_module_factory[train_mode]
    module_factory.generate_training_pipeline(args).main()
    
@PIPELINE_REGISTRY.register(suffix=__name__) # type: ignore
def prepare_inference(save_path):
    args: GTSEngineArgs = load_args(save_path) # type: ignore
    train_mode = TRAIN_MODE(args.train_mode)
    module_factory = mode_to_module_factory[train_mode]
    return module_factory.generate_inference_engine(args)

@PIPELINE_REGISTRY.register(suffix=__name__) # type: ignore
def inference(samples, inference_engine: BaseInferenceEngine):
    results = inference_engine.inference(sample)
    return results