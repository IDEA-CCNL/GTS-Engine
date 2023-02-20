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

from datasets import Dataset as hfDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class BagualuSummaryDataset(Dataset):
    """BagualuIEDataset.

    Args:
        data (hfDataset): data
        tokenizer (PreTrainedTokenizer): tokenizer
        args (TrainingArgumentsIEStd): arguments
    """

    def __init__(self, data: hfDataset, tokenizer: PreTrainedTokenizer,
                 max_enc_length: int, max_dec_length: int) -> None:
        super().__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.max_dec_length = max_dec_length
        self.max_enc_length = max_enc_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self._encode_on_iter(self.data[index], index)

    def _encode_on_iter(self, sample, idx: int):

        encode_dict = self.tokenizer.encode_plus(
            sample['text'],
            max_length=self.max_enc_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')

        decode_dict = self.tokenizer.encode_plus(
            sample['summary'],
            max_length=self.max_dec_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')

        labels = decode_dict['input_ids'].squeeze()  # type: ignore

        source_input = encode_dict['input_ids'].squeeze()  # type: ignore
        attn_mask = encode_dict['attention_mask'].squeeze()  # type: ignore

        return {
            'input_ids': source_input,
            'attention_mask': attn_mask,
            'labels': labels,
            'text': sample['text'],
            'summary': sample['summary'],
            'id': idx
        }
