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

# pylint: disable=no-member

import torch
from torch import nn, Tensor
from transformers import BertPreTrainedModel, BertModel, BertConfig


class Triaffine(nn.Module):
    """ Triaffine module

    Args:
        triaffine_hidden_size (int): Triaffine module hidden size
    """
    def __init__(self, triaffine_hidden_size: int) -> None:
        super().__init__()

        self.triaffine_hidden_size = triaffine_hidden_size

        self.weight_start_end = nn.Parameter(
            torch.zeros(triaffine_hidden_size,
                        triaffine_hidden_size,
                        triaffine_hidden_size))

        nn.init.normal_(self.weight_start_end, mean=0, std=0.1)

    def forward(self,
                start_logits: Tensor,
                end_logits: Tensor,
                cls_logits: Tensor) -> Tensor:
        """forward

        Args:
            start_logits (Tensor): start logits
            end_logits (Tensor): end logits
            cls_logits (Tensor): cls logits

        Returns:
            Tensor: span_logits
        """
        start_end_logits = torch.einsum("bxi,ioj,byj->bxyo",
                                        start_logits,
                                        self.weight_start_end,
                                        end_logits)

        span_logits = torch.einsum("bxyo,bzo->bxyz",
                                   start_end_logits,
                                   cls_logits)

        return span_logits


class MLPLayer(nn.Module):
    """MLP layer

    Args:
        input_size (int): input size
        output_size (int): output size
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor: # pylint: disable=invalid-name
        """ forward

        Args:
            x (Tensor): input

        Returns:
            Tensor: output
        """
        x = self.linear(x)
        x = self.act(x)
        return x


class BagualuIEModel(BertPreTrainedModel):
    """ BagualuIEModel

    Args:
        config (BertConfig): config
    """
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
        self.config = config

        self.triaffine_hidden_size = 128

        self.mlp_start = MLPLayer(self.config.hidden_size,
                                  self.triaffine_hidden_size)
        self.mlp_end = MLPLayer(self.config.hidden_size,
                                self.triaffine_hidden_size)
        self.mlp_cls = MLPLayer(self.config.hidden_size,
                                self.triaffine_hidden_size)

        self.triaffine = Triaffine(self.triaffine_hidden_size)

    def forward(self,  # pylint: disable=unused-argument
                input_ids: Tensor,
                attention_mask: Tensor,
                position_ids: Tensor,
                token_type_ids: Tensor,
                text_len: Tensor,
                label_token_idx: Tensor,
                **kwargs) -> Tensor:
        """ forward

        Args:
            input_ids (Tensor): input_ids
            attention_mask (Tensor): attention_mask
            position_ids (Tensor): position_ids
            token_type_ids (Tensor): token_type_ids
            text_len (Tensor): query length
            label_token_idx (Tensor, optional): label_token_idx

        Returns:
            Tensor: span logits
        """

        # bert forward
        hidden_states = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  position_ids=position_ids,
                                  token_type_ids=token_type_ids,
                                  output_hidden_states=True)[0]  # (bsz, seq, dim)

        max_text_len = text_len.max()

        # 获取start、end、cls的hidden_states
        hidden_start_end = hidden_states[:, :max_text_len, :] # text部分表示
        hidden_cls = hidden_states.gather(1, label_token_idx.unsqueeze(-1)\
            .repeat(1, 1, self.config.hidden_size)) # (bsz, task, dim)

        # Triaffine
        span_logits = self.triaffine(self.mlp_start(hidden_start_end),
                                     self.mlp_end(hidden_start_end),
                                     self.mlp_cls(hidden_cls)).sigmoid()

        return span_logits
