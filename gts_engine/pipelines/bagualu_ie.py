# coding=utf-8
# Copyright 2021 The IDEA Authors. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil
from typing import List, Literal, Tuple, Type, Callable, Dict, Any
from argparse import Namespace
from enum import Enum

from transformers import AutoModel, AutoTokenizer

from gts_common.registry import PIPELINE_REGISTRY
from gts_common.pipeline_utils import load_args

from bagualu.lib.framework.base_gts_engine_interface import BaseGtsEngineInterface, TRAIN_MODE, GtsEngineArgs
from bagualu.entrances.ie import GtsEngineInterfaceIEStd
from bagualu.entrances.ie.inference_manager_std import InferenceManagerIEStd

mode_to_interface: Dict[TRAIN_MODE, BaseGtsEngineInterface] = {
    TRAIN_MODE.STD: GtsEngineInterfaceIEStd()
}


def download_model_from_huggingface(pretrained_model_dir, model_name, model_class=AutoModel, tokenizer_class=AutoTokenizer):
    """ download """
    if os.path.exists(os.path.join(pretrained_model_dir, model_name)):
        print("model already exists.")
        return
    cache_path = os.path.join(pretrained_model_dir, "cache")
    model = model_class.from_pretrained(model_name, cache_dir=cache_path)
    tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=cache_path)
    model.save_pretrained(os.path.join(pretrained_model_dir, model_name))
    tokenizer.save_pretrained(os.path.join(pretrained_model_dir, model_name))
    shutil.rmtree(cache_path)
    print("model %s is downloaded from huggingface." % model_name)


@PIPELINE_REGISTRY.register(suffix=__name__)
def train_pipeline(args: GtsEngineArgs):
    """ pipeline """
    model_name = "hfl/chinese-macbert-base"
    download_model_from_huggingface(args.pretrained_model_dir, model_name)
    args.pretrained_model_dir = os.path.join(args.pretrained_model_dir, model_name)
    train_mode = TRAIN_MODE(args.train_mode)
    module_factory: GtsEngineInterfaceIEStd = mode_to_interface[train_mode]
    module_factory.prepare_training(args)
    module_factory.generate_training_pipeline(args).main()


@PIPELINE_REGISTRY.register(suffix=__name__)
def prepare_inference(save_path):
    """ pipeline """
    model_name = "hfl/chinese-macbert-base"
    args: GtsEngineArgs = load_args(save_path)
    download_model_from_huggingface(args.pretrained_model_dir, model_name)
    args.pretrained_model_dir = os.path.join(args.pretrained_model_dir, model_name)
    train_mode = TRAIN_MODE(args.train_mode)
    module_factory: GtsEngineInterfaceIEStd = mode_to_interface[train_mode]
    return module_factory.generate_inference_manager(args)


@PIPELINE_REGISTRY.register(suffix=__name__)
def inference(samples: List[Dict[str, Any]], inference_manager: InferenceManagerIEStd):
    """ inference """
    results = inference_manager.inference(samples)
    return results
