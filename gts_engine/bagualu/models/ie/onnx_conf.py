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


class BagualuIEOnnxConfig(object):
    """ config """
    def __init__(self) -> None:
        self.input_names = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "token_type_ids",
            "text_len",
            "label_token_idx",
        ]
        self.output_names = [
            "span_logits"
        ]
        self.dynamic_axes = {

            # input
            "input_ids": {
                0: "batch_size",
                1: "input_len",
            },
            "attention_mask": {
                0: "batch_size",
                1: "input_len",
                2: "input_len",
            },
            "position_ids": {
                0: "batch_size",
                1: "input_len",
            },
            "token_type_ids": {
                0: "batch_size",
                1: "input_len",
            },
            "text_len": {
                0: "batch_size",
            },
            "label_token_idx": {
                0: "batch_size",
                1: "num_labels",
            },

            # output
            "span_logits": {
                0: "batch_size",
                1: "seq_len",
                2: "seq_len",
                3: "num_labels",
            },
        }
