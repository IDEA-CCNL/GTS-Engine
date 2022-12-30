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
from typing import List, Dict, Any

from gts_common.registry import PIPELINE_REGISTRY
from gts_common.pipeline_utils import load_args, save_args, download_model_from_huggingface
from gts_common.arguments import GtsEngineArgs

from bagualu.lib.framework.base_gts_engine_interface import BaseGtsEngineInterface, TRAIN_MODE
from bagualu.entrances.ie import GtsEngineInterfaceIEStd
from bagualu.entrances.ie.inference_manager_std import InferenceManagerIEStd
from bagualu.models.ie import BagualuIEModel

mode_to_interface: Dict[TRAIN_MODE, BaseGtsEngineInterface] = {
    TRAIN_MODE.STD: GtsEngineInterfaceIEStd()
}


@PIPELINE_REGISTRY.register(suffix=__name__)
def train_pipeline(args: GtsEngineArgs) -> None:
    """ Bagualu IE training pipeline

    Args:
        args (GtsEngineArgs): user arguments
    """
    # save args
    args = save_args(args)
    model_name = "Erlangshen-BERT-120M-IE-Chinese"
    download_model_from_huggingface(args.pretrained_model_dir,
                                    model_name,
                                    model_class=BagualuIEModel)
    args.pretrained_model_dir = os.path.join(args.pretrained_model_dir, model_name)
    train_mode = TRAIN_MODE(args.train_mode)
    module_factory: GtsEngineInterfaceIEStd = mode_to_interface[train_mode]
    module_factory.prepare_training(args)
    module_factory.generate_training_pipeline(args).main()


@PIPELINE_REGISTRY.register(suffix=__name__)
def prepare_inference(save_path: str) -> InferenceManagerIEStd:
    """ prepare inference

    Args:
        save_path (str): saved path from training.

    Returns:
        InferenceManagerIEStd: inference manager.
    """
    model_name = "Erlangshen-BERT-120M-IE-Chinese"
    args: GtsEngineArgs = load_args(save_path)
    download_model_from_huggingface(args.pretrained_model_dir,
                                    model_name,
                                    model_class=BagualuIEModel)
    args.pretrained_model_dir = os.path.join(args.pretrained_model_dir, model_name)
    train_mode = TRAIN_MODE(args.train_mode)
    module_factory: GtsEngineInterfaceIEStd = mode_to_interface[train_mode]
    return module_factory.generate_inference_manager(args)


@PIPELINE_REGISTRY.register(suffix=__name__)
def inference(samples: List[Dict[str, Any]],
              inference_manager: InferenceManagerIEStd) -> List[dict]:
    """ inference

    Args:
        samples (List[Dict[str, Any]]): input samples for inference.
        inference_manager (InferenceManagerIEStd): inference manager.

    Returns:
        List[dict]: inference results
    """
    results = inference_manager.inference(samples)
    return results
