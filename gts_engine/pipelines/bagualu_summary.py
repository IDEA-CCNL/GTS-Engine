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
from typing import Any, Dict, List

from bagualu.entrances.summary import GtsEngineInterfaceSummaryStd
from bagualu.entrances.summary.inference_manager_std import \
    InferenceManagerSummaryStd
from bagualu.models.summary import PegasusTokenizer
from gts_common.arguments import GtsEngineArgs
from gts_common.framework.base_gts_engine_interface import (
    TRAIN_MODE, BaseGtsEngineInterface)
from gts_common.pipeline_utils import (download_model_from_huggingface,
                                       load_args, save_args)
from gts_common.registry import PIPELINE_REGISTRY
from transformers.models.pegasus.modeling_pegasus import \
    PegasusForConditionalGeneration

mode_to_interface: Dict[TRAIN_MODE, BaseGtsEngineInterface] = {
    TRAIN_MODE.STD: GtsEngineInterfaceSummaryStd(),
    TRAIN_MODE.ADV: GtsEngineInterfaceSummaryStd()
}


@PIPELINE_REGISTRY.register(suffix=__name__)
def train_pipeline(args: GtsEngineArgs) -> None:
    """Bagualu Summary training pipeline.

    Args:
        args (GtsEngineArgs): user arguments
    """
    # save args
    args = save_args(args)
    if args.train_mode == "standard":
        model_name = "Randeng-Pegasus-238M-Summary-Chinese"
    elif args.train_mode == "advanced":
        model_name = "Randeng-Pegasus-523M-Summary-Chinese"

    download_model_from_huggingface(
        args.pretrained_model_dir,
        model_name,
        model_class=PegasusForConditionalGeneration,
        tokenizer_class=PegasusTokenizer)
    args.pretrained_model_dir = os.path.join(args.pretrained_model_dir,
                                             model_name)
    train_mode = TRAIN_MODE(args.train_mode)
    module_factory: GtsEngineInterfaceSummaryStd = mode_to_interface[
        train_mode]
    module_factory.prepare_training(args)
    module_factory.generate_training_pipeline(args).main()


@PIPELINE_REGISTRY.register(suffix=__name__)
def prepare_inference(save_path: str) -> InferenceManagerSummaryStd:
    """prepare inference.

    Args:
        save_path (str): saved path from training.

    Returns:
        InferenceManagerSummaryStd: inference manager.
    """
    model_name = "Randeng-Pegasus-238M-Summary-Chinese"
    args: GtsEngineArgs = load_args(save_path)
    download_model_from_huggingface(
        args.pretrained_model_dir,
        model_name,
        model_class=PegasusForConditionalGeneration,
        tokenizer_class=PegasusTokenizer)
    args.pretrained_model_dir = os.path.join(args.pretrained_model_dir,
                                             model_name)
    train_mode = TRAIN_MODE(args.train_mode)
    module_factory: GtsEngineInterfaceSummaryStd = mode_to_interface[
        train_mode]
    return module_factory.generate_inference_manager(args)


@PIPELINE_REGISTRY.register(suffix=__name__)
def inference(samples: List[Dict[str, Any]],
              inference_manager: InferenceManagerSummaryStd) -> List[dict]:
    """inference.

    Args:
        samples (List[Dict[str, Any]]): input samples for inference.
        inference_manager (InferenceManagerSummaryStd): inference manager.

    Returns:
        List[dict]: inference results
    """
    results = inference_manager.inference(samples)
    return results
