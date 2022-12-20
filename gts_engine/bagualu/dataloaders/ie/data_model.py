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

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import PreTrainedTokenizer

from .item_encoder import ItemEncoder
from .dataset import BagualuIEDataset
from ...arguments.ie import TrainingArgumentsIEStd


class BagualuIEDataModel(pl.LightningDataModule):
    """ uniEXDataModel

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer
        args (GTSEngineArgs): arguments
        train_data (List[dict]): train data
        dev_data (List[dict]): dev data
        test_data (List[dict]): test data
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 args: TrainingArgumentsIEStd,
                 train_data: List[dict] = None,
                 dev_data: List[dict] = None,
                 test_data: List[dict] = None):
        super().__init__()

        self.batch_size = args.batch_size

        self.train_data = BagualuIEDataset(train_data if train_data else [], tokenizer, args.max_length)
        self.valid_data = BagualuIEDataset(dev_data if dev_data else [], tokenizer, args.max_length)
        self.test_data = BagualuIEDataset(test_data if test_data else [], tokenizer, args.max_length)

        if not args.distill_self:
            self.collate_fn = ItemEncoder.collate
        else:
            self.collate_fn = ItemEncoder.collate_expand

    def train_dataloader(self) -> DataLoader:
        """ train_dataloader

        Returns:
            DataLoader: data loader
        """
        return DataLoader(self.train_data,
                          shuffle=True,
                          collate_fn=self.collate_fn,
                          batch_size=self.batch_size,
                          pin_memory=False)

    def val_dataloader(self) -> DataLoader:
        """ val_dataloader

        Returns:
            DataLoader: data loader
        """
        return DataLoader(self.valid_data,
                          shuffle=False,
                          collate_fn=self.collate_fn,
                          batch_size=self.batch_size,
                          pin_memory=False)

    def test_dataloader(self) -> DataLoader:
        """ test_dataloader

        Returns:
            DataLoader: data loader
        """
        return DataLoader(self.test_data,
                          shuffle=False,
                          collate_fn=self.collate_fn,
                          batch_size=self.batch_size,
                          pin_memory=False)
