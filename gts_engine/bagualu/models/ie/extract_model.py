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
import copy

from transformers import PreTrainedTokenizer

from ...dataloaders.ie.item_encoder import ItemEncoder
from ...dataloaders.ie.item_decoder import ItemDecoder
from ...arguments.ie import TrainingArgumentsIEStd
from .model import BagualuIEModel


class BagualuIEExtractModel(object):
    """ BagualuIEExtractModel

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer
        args (TrainingArgumentsIEStd): arguments
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 args: TrainingArgumentsIEStd) -> None:
        self.encoder = ItemEncoder(tokenizer, args.max_length)
        self.decoder = ItemDecoder(tokenizer, args)

    def extract(self, batch_data: List[dict], model: BagualuIEModel) -> List[dict]:
        """ extract

        Args:
            batch_data (List[dict]): batch of data
            model (BagualuIEModel): model

        Returns:
            List[dict]: batch of data
        """

        model = model.cuda()
        model.eval()

        batch_data = copy.deepcopy(batch_data)
        batch = [self.encoder.encode_item(item, with_label=False) for item in batch_data]
        batch = self.encoder.collate(batch)

        batch = {k: v.cuda() for k, v in batch.items()}

        span_logits = model(**batch).cpu().detach().numpy()
        label_mask = batch["label_mask"].cpu().detach().numpy()

        for i, item in enumerate(batch_data):

            entity_list, spo_list = self.decoder.decode(item,
                                                        span_logits[i],
                                                        label_mask[i])

            item["spo_list"] = spo_list
            item["entity_list"] = entity_list

        return batch_data
