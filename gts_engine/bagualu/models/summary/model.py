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

# pylint: disable=no-member

import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, KLDivLoss
from transformers.models.pegasus.modeling_pegasus import \
    PegasusForConditionalGeneration


class BagualuSummaryModel(nn.Module):
    """BagualuSummaryModel.

    Args:
        config (BertConfig): config
    """

    def __init__(self, pretrained_model_dir: str, max_dec_length: int) -> None:
        super().__init__()

        self._model = PegasusForConditionalGeneration.from_pretrained(
            pretrained_model_dir)
        self._config = self._model.config  # type: ignore

        self._CELoss = CrossEntropyLoss()
        self._KLLoss = KLDivLoss(reduction='batchmean')

        self._max_dec_length = max_dec_length

    def forward(
            self,  # pylint: disable=unused-argument
            input_ids: Tensor,
            attention_mask: Tensor,
            labels: Tensor,
            is_training=True,
            **kwargs) -> Tensor:
        """forward.

        Args:
            input_ids (Tensor): input_ids
            attention_mask (Tensor): attention_mask
            labels (Tensor, optional): labels

        Returns:
            Tensor: loss
            Tensor: generated_ids
        """

        outputs = self._model.forward(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      labels=labels,
                                      return_dict=True)
        if is_training:
            generated_ids = None
        else:

            generated_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self._max_dec_length,
                num_beams=8,
            )  # type: ignore

        masked_lm_loss: float = 0.
        masked_lm_loss = self._CELoss(
            outputs.logits.view(-1, self._config.vocab_size), labels.view(-1))

        kl_loss: float = 0.
        # if soft_labels is not None:
        #     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        #     kl_loss = self._KLLoss(probs.log(), soft_labels)

        loss_total: float = masked_lm_loss + kl_loss

        return {'loss_total': loss_total, 'generated_ids': generated_ids}
