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

from gts_common.framework import BaseInferenceManager
from gts_common.framework.mixin import OptionalLoggerMixin
from transformers import AutoTokenizer

from ...arguments.summary import InferenceArgumentsSummaryStd
from ...dataloaders.summary import BagualuSummaryDataModule
from ...models.summary import (BagualuSummaryLitModel, BagualuSummaryModel,
                               PegasusTokenizer)


class InferenceManagerSummaryStd(BaseInferenceManager, OptionalLoggerMixin):
    """InferenceManagerSummaryStd."""

    _args: InferenceArgumentsSummaryStd  # inference所需参数
    _inference_model: BagualuSummaryModel  # inference模型

    def prepare_inference(self) -> None:
        """prepare inference."""
        # load model
        self._inference_model = BagualuSummaryLitModel.load_from_checkpoint(
            self._args.best_ckpt_path,  # pylint: disable=protected-access
            args=self._args,
            logger=None)._model
        self.info(f"loaded model from {self._args.best_ckpt_path}")

        # load tokenizer
        tokenizer = PegasusTokenizer.from_pretrained(
            self._args.pretrained_model_root)
        self.info(f"loaded tokenzier from {self._args.pretrained_model_root}")
        self._tokenizer = tokenizer

    def inference(self, sample: List[dict]) -> dict:
        """inference.

        Args:
            sample (List[dict]): input samples for inference

        Returns:
            List[dict]: inference results
        """

        inf_data_module = BagualuSummaryDataModule(self._tokenizer, self._args,
                                                   None, None, sample)

        data_loader = inf_data_module.test_dataloader()

        result = dict()
        for batch in data_loader:

            model_output = self._inference_model(**batch, is_training=False)
            preds = self._tokenizer.batch_decode(
                model_output['generated_ids'],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True)

            for idx, pred in zip(batch['id'], preds):
                result[idx] = pred

        return result
