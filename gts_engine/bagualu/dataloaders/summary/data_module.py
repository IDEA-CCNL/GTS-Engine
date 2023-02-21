# Copyright 2021 The IDEA Authors. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

import os
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List

import pytorch_lightning as pl
import torch
from datasets import Dataset as hfDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from ...arguments.summary import TrainingArgumentsSummaryStd
from ...models.summary.tokenizers_pegasus import PegasusTokenizer
from .dataset import BagualuSummaryDataset


class BagualuSummaryDataModule(pl.LightningDataModule):
    """BagualuSummaryDataModule.

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer
        args (GTSEngineArgs): arguments
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 args: TrainingArgumentsSummaryStd,
                 train_data: hfDataset = None,
                 dev_data: hfDataset = None,
                 test_data: hfDataset = None):
        super().__init__()

        self.batch_size = args.batch_size
        self.tokenizer = tokenizer
        self.num_workers = args.num_workers

        self.train_data = BagualuSummaryDataset(
            train_data if train_data else [], tokenizer, args.max_enc_length,
            args.max_enc_length)
        self.valid_data = BagualuSummaryDataset(dev_data if dev_data else [],
                                                tokenizer, args.max_enc_length,
                                                args.max_enc_length)
        self.test_data = BagualuSummaryDataset(test_data if test_data else [],
                                               tokenizer, args.max_enc_length,
                                               args.max_enc_length)

    def train_dataloader(self) -> DataLoader:
        """train_dataloader.

        Returns:
            DataLoader: data loader
        """

        return DataLoader(self.train_data,
                          shuffle=True,
                          collate_fn=self.dynamic_padding_fn,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False)

    def val_dataloader(self) -> DataLoader:
        """val_dataloader.

        Returns:
            DataLoader: data loader
        """
        return DataLoader(self.valid_data,
                          shuffle=False,
                          collate_fn=self.dynamic_padding_fn,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False)

    def test_dataloader(self) -> DataLoader:
        """test_dataloader.

        Returns:
            DataLoader: data loader
        """
        return DataLoader(self.test_data,
                          shuffle=False,
                          collate_fn=self.dynamic_padding_fn,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=False)

    def dynamic_padding_fn(self, batch: List[Dict]) -> Dict:
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]

        input_ids = torch.stack(batch_data['input_ids'])
        attention_mask = torch.stack(batch_data['attention_mask'])
        # decoder_input_ids = torch.stack(batch_data['decoder_input_ids'])
        # decoder_attention_mask = torch.stack(batch_data['decoder_attention_mask'])
        summary = batch_data['summary'] if 'summary' in batch_data else []

        # decoder_input_ids = shift_tokens_right(decoder_input_ids, self._tokenizer.pad_token_id, self._tokenizer.pad_token_id) # type: ignore
        # decoder_input_ids = decoder_input_ids[:,:-1] # remove <eos>

        if 'labels' in batch_data:
            labels = torch.stack(batch_data['labels'])
            end_token_index = torch.where(
                labels == self.tokenizer.eos_token_id)[1]
            for idx, end_idx in enumerate(end_token_index):
                labels[idx][end_idx + 1:] = -100
        else:
            labels = None

        batch_data = {
                'id': batch_data['id'],
                'text': batch_data['text'],
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        
        if labels is not None:
            batch_data["summary"] = summary
            batch_data["labels"] = labels


        return batch_data
