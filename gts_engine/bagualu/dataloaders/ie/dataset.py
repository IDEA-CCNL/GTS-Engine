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

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .item_encoder import ItemEncoder
from ...arguments.ie import TrainingArgumentsIEStd


class UniEXDataset(Dataset):
    """ UniEXDataset

    Args:
        data (List[dict]): data
        tokenizer (PreTrainedTokenizer): tokenizer
        args (TrainingArgumentsIEStd): arguments
        used_mask (bool, optional): used_mask. Defaults to False.
    """
    def __init__(self,
                 data: List[dict],
                 tokenizer: PreTrainedTokenizer,
                 args: TrainingArgumentsIEStd) -> None:
        super().__init__()

        self.data = data
        self.encoder = ItemEncoder(tokenizer, args.max_length)
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.encoder.encode_item(self.data[index])
