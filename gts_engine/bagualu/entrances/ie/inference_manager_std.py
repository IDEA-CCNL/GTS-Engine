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

from typing import List

from transformers import AutoTokenizer

from ...lib.framework import BaseInferenceManager
from ...lib.framework.mixin import OptionalLoggerMixin
from ...models.ie import BagualuIEModel, BagualuIELitModel, BagualuIEExtractModel
from ...arguments.ie import InferenceArgumentsIEStd
from ...dataloaders.ie import data_segment, data_segment_restore


class InferenceManagerIEStd(BaseInferenceManager, OptionalLoggerMixin):
    """ InferenceManagerIEStd """

    _args: InferenceArgumentsIEStd # inference所需参数
    _inference_model: BagualuIEModel # inference模型
    _extract_model: BagualuIEExtractModel # 抽取模型

    def prepare_inference(self) -> None:
        """ prepare inference """
        # load model
        self._inference_model = BagualuIELitModel.load_from_checkpoint(self._args.best_ckpt_path, # pylint: disable=protected-access
                                                                       args=self._args,
                                                                       logger=None)._model
        self.info(f"loaded model from {self._args.best_ckpt_path}")

        # load tokenizer
        added_token = [f"[unused{i + 1}]" for i in range(99)]
        tokenizer = AutoTokenizer.from_pretrained(self._args.model_save_dir,
                                                  additional_special_tokens=added_token)
        self.info(f"loaded tokenzier from {self._args.model_save_dir}")

        # instantiate extract model
        self._extract_model = BagualuIEExtractModel(tokenizer, self._args)

    def inference(self, sample: List[dict]) -> List[dict]:
        """ inference

        Args:
            sample (List[dict]): input samples for inference

        Returns:
            List[dict]: inference results
        """
        sample = data_segment(sample)
        batch_size = self._args.batch_size
        result = []
        for i in range(0, len(sample), batch_size):
            batch_data = sample[i: i + batch_size]
            batch_result = self._extract_model.extract(batch_data, self._inference_model)
            result.extend(batch_result)
        result = data_segment_restore(result)
        return result
