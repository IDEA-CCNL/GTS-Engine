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

from typing import List, Tuple, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from .dataset_utils import get_choice


def get_entity_indices(entity_list: List[dict], spo_list: List[dict]) -> List[List[int]]:
    """ 获取样本中包含的实体位置信息

    Args:
        entity_list (List[dict]): 实体列表
        spo_list (List[dict]): 三元组列表

    Returns:
        List[List[int]]: 实体位置信息
    """
    entity_indices = []

    # 实体中的实体位置
    for entity in entity_list:
        entity_index = entity["entity_index"]
        entity_indices.append(entity_index)

    # 三元组中的实体位置
    for spo in spo_list:
        sub_idx = spo["subject"]["entity_index"]
        obj_idx = spo["object"]["entity_index"]
        entity_indices.append(sub_idx)
        entity_indices.append(obj_idx)

    return entity_indices


def entity_based_tokenize(text: str,
                          tokenizer: PreTrainedTokenizer,
                          enitity_indices: List[Tuple[int, int]],
                          max_len: int = -1,
                          return_offsets_mapping: bool = False) \
    -> Union[List[int], Tuple[List[int], List[Tuple[int, int]]]]:
    """ 基于实体位置信息的编码，确保实体为连续1到多个token的合并，同时利用预训练模型词根信息

    Args:
        text (str): 文本
        tokenizer (PreTrainedTokenizer): tokenizer
        enitity_indices (List[Tuple[int, int]]): 实体位置信息
        max_len (int, optional): 长度限制. Defaults to -1.
        return_offsets_mapping (bool, optional): 是否返回offsets_mapping. Defaults to False.

    Returns:
        Union[List[int], Tuple[List[int], List[Tuple[int, int]]]]: 编码id
    """
    # 根据实体位置遍历出需要对文本进行切割的点
    split_points = sorted(list({i for idx in enitity_indices for i in idx} | {0, len(text)}))
    # 对文本进行切割
    text_parts = []
    for i in range(0, len(split_points) - 1):
        text_parts.append(text[split_points[i]: split_points[i + 1]])

    # 对切割后的文本进行编码
    bias = 0
    text_ids = []
    offset_mapping = []
    for part in text_parts:

        part_encoded = tokenizer(part, add_special_tokens=False, return_offsets_mapping=True)
        part_ids, part_mapping = part_encoded["input_ids"], part_encoded["offset_mapping"]

        text_ids.extend(part_ids)
        for start, end in part_mapping:
            offset_mapping.append((start + bias, end + bias))

        bias += len(part)

    if max_len > 0:
        text_ids = text_ids[: max_len]

    # 是否返回offsets_mapping
    if return_offsets_mapping:
        return text_ids, offset_mapping
    return text_ids


class ItemEncoder(object):
    """ Item Encoder

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer
        max_length (int): max length
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def search_index(self,
                     entity_idx: List[int],
                     offset_mapping: List[Tuple[int, int]],
                     bias: int = 0) -> Tuple[int, int]:
        """ 查找实体在tokens中的索引

        Args:
            entity_idx (List[int]): entity index
            offset_mapping (List[Tuple[int, int]]): text
            bias (int): bias

        Returns:
            Tuple[int]: (start_idx, end_idx)
        """
        entity_start, entity_end = entity_idx
        start_idx, end_idx = -1, -1

        for token_idx, (start, end) in enumerate(offset_mapping):
            if start == entity_start:
                start_idx = token_idx
            if end == entity_end:
                end_idx = token_idx
        assert start_idx >= 0 and end_idx >= 0

        return start_idx + bias, end_idx + bias

    @staticmethod
    def get_position_ids(text_len: int,
                        ent_ranges: List,
                        rel_ranges: List) -> np.ndarray:
        """ 获取position_ids

        Args:
            text_len (int): input length
            ent_ranges (List[List[int, int]]): each entity ranges idx
            rel_ranges (List[List[int, int]]): each relation ranges idx.

        Returns:
            np.ndarray: position_ids
        """
        # 一切从0开始算position，@liuhan
        text_pos_ids = list(range(text_len))

        ent_pos_ids, rel_pos_ids = [], []
        for s, e in ent_ranges:
            ent_pos_ids.extend(list(range(e - s)))
        for s, e in rel_ranges:
            rel_pos_ids.extend(list(range(e - s)))
        position_ids = text_pos_ids + ent_pos_ids + rel_pos_ids

        return position_ids

    @staticmethod
    def get_att_mask(input_len: int,
                     ent_ranges: List,
                     rel_ranges: List= None,
                     choice_ent: List[str] = None,
                     choice_rel: List[str] = None,
                     entity2rel: dict = None,
                     full_attent: bool = False) -> np.ndarray:
        """ 获取att_mask，不同choice之间的attention_mask置零

        Args:
            input_len (int): input length
            ent_ranges (List[List[int, int]]): each entity ranges idx
            rel_ranges (List[List[int, int]]): each relation ranges idx. Defaults to None.
            choice_ent (List[str], optional): choice entity. Defaults to None.
            choice_rel (List[str], optional): choice relation. Defaults to None.
            entity2rel (dict, optional): entity to relations. Defaults to None.
            full_attent (bool, optional): is full attention or not. Defaults to None.
        Returns:
            np.ndarray: attention mask
        """

        # attention_mask.shape = (input_len, input_len)
        attention_mask = np.ones((input_len, input_len))
        if full_attent and not rel_ranges: # full-attention且没有关系情况下，返回全1
            return attention_mask

        # input_ids: [CLS] text [SEP] [unused1] ent1 [unused2] rel1 [unused3] event1
        text_len = ent_ranges[0][0] # text长度
        # 将text-实体之间的attention置零，text看不到实体,不受传入的entity个数、顺序影响 @liuhan
        attention_mask[:text_len, text_len:] = 0

        # 将实体-实体、实体关系之间的attention_mask置零
        attention_mask[text_len:, text_len: ] = 0

        # 将每个实体与自己的attention_mask置一
        for s, e in ent_ranges:
            attention_mask[s: e, s: e] = 1

        # 没有关系的话，直接返回
        if not rel_ranges:
            return attention_mask

        # 处理有关系情况

        # 关系自身attention_mask置1
        for s, e in rel_ranges:
            attention_mask[s: e, s: e] = 1

        # 将有关联的实体-关系置一
        for head_tail, relations in entity2rel.items():
            for entity_type in head_tail:
                ent_idx = choice_ent.index(entity_type)
                ent_s, _ = ent_ranges[ent_idx] # ent_s, ent_e
                for relation_type in relations:
                    rel_idx = choice_rel.index(relation_type)
                    rel_s, rel_e = rel_ranges[rel_idx]
                    attention_mask[rel_s: rel_e, ent_s] = 1 # 关系只看实体第一个的[unused1]

        if full_attent: # full-attention且有关系情况下，让文本能看见关系
            for s, e in rel_ranges:
                attention_mask[: text_len, s: e] = 1

        return attention_mask

    def encode(self,
               text: str,
               task_name: str,
               choice: List[str],
               entity_list: List[dict],
               spo_list: List[dict],
               full_attent: bool = False,
               with_label: bool = True) -> Dict[str, torch.Tensor]:
        """ encode

        Args:
            text (str): text
            task_name (str): task name
            choice (List[str]): choice
            entity_list (List[dict]): entity list
            spo_list (List[dict]): spo list
            full_attent (bool): full attention
            with_label (bool): encoded with label. Defaults to True.

        Returns:
            Dict[str, torch.Tensor]: encoded
        """
        choice_ent, choice_rel, entity2rel = choice, [], {}
        if isinstance(choice, list):
            if isinstance(choice[0], list): # 关系抽取 & 实体识别
                choice_ent, choice_rel, _, _, entity2rel = get_choice(choice)
        elif isinstance(choice, dict):
            # 事件类型
            raise ValueError('event extract not supported now!')
        else:
            raise NotImplementedError

        input_ids = []
        text_ids = [] # text部分id
        ent_ids = [] # entity部分id
        rel_ids = [] # relation部分id
        entity_labels_idx = []
        relation_labels_idx = []

        sep_ids = self.tokenizer.encode("[SEP]", add_special_tokens=False) # [SEP]的编码
        cls_ids = self.tokenizer.encode("[CLS]", add_special_tokens=False) # [CLS]的编码
        entity_op_ids = self.tokenizer.encode("[unused1]", add_special_tokens=False) # [unused1]的编码
        relation_op_ids = self.tokenizer.encode("[unused2]", add_special_tokens=False) # [unused2]的编码

        # 任务名称的编码
        task_ids = self.tokenizer.encode(task_name, add_special_tokens=False)

        # 实体标签的编码
        for c in choice_ent:
            c_ids = self.tokenizer.encode(c, add_special_tokens=False)[: self.max_length]
            ent_ids += entity_op_ids + c_ids

        # 关系标签的编码
        for c in choice_rel:
            c_ids = self.tokenizer.encode(c, add_special_tokens=False)[: self.max_length]
            rel_ids += relation_op_ids + c_ids

        # text的编码
        entity_indices = get_entity_indices(entity_list, spo_list)
        text_max_len = self.max_length - len(task_ids) - 3
        text_ids, offset_mapping = entity_based_tokenize(text, self.tokenizer, entity_indices,
                                                         max_len=text_max_len,
                                                         return_offsets_mapping=True)
        text_ids = cls_ids + text_ids + sep_ids

        input_ids = text_ids + task_ids + sep_ids + ent_ids + rel_ids

        token_type_ids = [0] * len(text_ids) + [0] * (len(task_ids) + 1) + \
            [1] * len(ent_ids) + [1] * len(rel_ids)

        entity_labels_idx = [i for i, id_ in enumerate(input_ids) if id_ == entity_op_ids[0]]
        relation_labels_idx = [i for i, id_ in enumerate(input_ids) if id_ == relation_op_ids[0]]

        ent_ranges = [] # 每个实体的起始范围
        for i in range(len(entity_labels_idx) - 1):
            ent_ranges.append([entity_labels_idx[i], entity_labels_idx[i + 1]])
        if not relation_labels_idx:
            ent_ranges.append([entity_labels_idx[-1], len(input_ids)])
        else:
            ent_ranges.append([entity_labels_idx[-1], relation_labels_idx[0]])
        assert len(ent_ranges) == len(choice_ent)

        rel_ranges = [] # 每个关系的起始范围
        for i in range(len(relation_labels_idx) - 1):
            rel_ranges.append([relation_labels_idx[i], relation_labels_idx[i + 1]])
        if relation_labels_idx:
            rel_ranges.append([relation_labels_idx[-1], len(input_ids)])
        assert len(rel_ranges) == len(choice_rel)

        # 所有unused的位置
        label_token_idx = entity_labels_idx + relation_labels_idx
        task_num_labels = len(label_token_idx)
        input_len = len(input_ids)
        text_len = len(text_ids)

        # 计算mask
        attention_mask = self.get_att_mask(input_len,
                                           ent_ranges,
                                           rel_ranges,
                                           choice_ent,
                                           choice_rel,
                                           entity2rel,
                                           full_attent)
        # 计算label-mask
        label_mask = np.ones((text_len, text_len, task_num_labels))
        for i in range(text_len):
            for j in range(text_len):
                if j < i:
                    for l in range(len(entity_labels_idx)):
                        # entity部分的下三角可mask
                        label_mask[i, j, l] = 0

        # 计算position_ids
        position_ids = self.get_position_ids(len(text_ids) + len(task_ids) + 1,
                                             ent_ranges,
                                             rel_ranges)

        assert len(input_ids) == len(position_ids) == len(token_type_ids)

        if not with_label:
            return {
                "input_ids": torch.tensor(input_ids).long(),
                "attention_mask": torch.tensor(attention_mask).float(),
                "position_ids": torch.tensor(position_ids).long(),
                "token_type_ids": torch.tensor(token_type_ids).long(),
                "label_token_idx": torch.tensor(label_token_idx).long(),
                "label_mask":  torch.tensor(label_mask).float(),
                "text_len": torch.tensor(text_len).long(),
                "ent_ranges": ent_ranges,
                "rel_ranges": rel_ranges,
            }

        # 输入的span_labels，只保留text部分
        span_labels = np.zeros((text_len, text_len, task_num_labels))

        # 将实体转成span
        for entity in entity_list:

            entity_type = entity["entity_type"]
            entity_index = entity["entity_index"]

            start_idx, end_idx = self.search_index(entity_index, offset_mapping, 1)

            if start_idx < text_len and end_idx < text_len:
                ent_label = choice_ent.index(entity_type)
                span_labels[start_idx, end_idx, ent_label] = 1

        # 将三元组转成span
        for spo in spo_list:

            sub_idx = spo["subject"]["entity_index"]
            obj_idx = spo["object"]["entity_index"]

            # 获取头实体、尾实体的开始、结束index
            sub_start_idx, sub_end_idx = self.search_index(sub_idx, offset_mapping, 1)
            obj_start_idx, obj_end_idx = self.search_index(obj_idx, offset_mapping, 1)
            # 实体label置1
            if sub_start_idx < text_len and sub_end_idx < text_len:
                sub_label = choice_ent.index(spo["subject"]["entity_type"])
                span_labels[sub_start_idx, sub_end_idx, sub_label] = 1

            if obj_start_idx < text_len and obj_end_idx < text_len:
                obj_label = choice_ent.index(spo["object"]["entity_type"])
                span_labels[obj_start_idx, obj_end_idx, obj_label] = 1

            # 有关系的sub/obj实体的start/end在realtion对应的label置1
            if spo["predicate"] in choice_rel:
                pre_label = choice_rel.index(spo["predicate"]) + len(choice_ent)
                if sub_start_idx < text_len and obj_start_idx < text_len:
                    span_labels[sub_start_idx, obj_start_idx, pre_label] = 1
                if sub_end_idx < text_len and obj_end_idx < text_len:
                    span_labels[sub_end_idx, obj_end_idx, pre_label] = 1

        return {
            "input_ids": torch.tensor(input_ids).long(),
            "attention_mask": torch.tensor(attention_mask).float(),
            "position_ids": torch.tensor(position_ids).long(),
            "token_type_ids": torch.tensor(token_type_ids).long(),
            "label_token_idx": torch.tensor(label_token_idx).long(),
            "span_labels": torch.tensor(span_labels).float(),
            "label_mask":  torch.tensor(label_mask).float(),
            "text_len": torch.tensor(text_len).long(),
            "ent_ranges": ent_ranges,
            "rel_ranges": rel_ranges,
        }

    def encode_item(self, item: dict, with_label: bool = True) -> Dict[str, torch.Tensor]:  # pylint: disable=unused-argument
        """ encode

        Args:
            item (dict): item
            with_label (bool): encoded with label. Defaults to True.

        Returns:
            Dict[str, torch.Tensor]: encoded
        """
        return self.encode(text=item["text"],
                           task_name=item["task"],
                           choice=item["choice"],
                           entity_list=item.get("entity_list", []),
                           spo_list=item.get("spo_list", []),
                           full_attent=item.get('full_attent', False),
                           with_label=with_label)

    @staticmethod
    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate a batch data.
        batch = [ins1_dict, ins2_dict, ..., insN_dict]
        batch_data = {"sentence":[ins1_sentence, ins2_sentence...],
        "input_ids":[ins1_input_ids, ins2_input_ids...], ...}
        """
        input_ids = nn.utils.rnn.pad_sequence(
            sequences=[encoded["input_ids"] for encoded in batch],
            batch_first=True,
            padding_value=0)

        label_token_idx = nn.utils.rnn.pad_sequence(
            sequences=[encoded["label_token_idx"] for encoded in batch],
            batch_first=True,
            padding_value=0)

        token_type_ids = nn.utils.rnn.pad_sequence(
            sequences=[encoded["token_type_ids"] for encoded in batch],
            batch_first=True,
            padding_value=0)

        position_ids = nn.utils.rnn.pad_sequence(
            sequences=[encoded["position_ids"] for encoded in batch],
            batch_first=True,
            padding_value=0)

        text_len = torch.tensor([encoded["text_len"] for encoded in batch]).long()
        max_text_len = text_len.max()

        batch_size, batch_max_length = input_ids.shape
        _, batch_max_labels = label_token_idx.shape

        attention_mask = torch.zeros((batch_size, batch_max_length, batch_max_length))
        label_mask = torch.zeros((batch_size,
                                  batch_max_length,
                                  batch_max_length,
                                  batch_max_labels))
        for i, encoded in enumerate(batch):
            input_len = encoded["attention_mask"].shape[0]
            attention_mask[i, :input_len, :input_len] = encoded["attention_mask"]
            _, cur_text_len, label_len = encoded['label_mask'].shape
            label_mask[i, :cur_text_len, :cur_text_len, :label_len] = encoded['label_mask']
        label_mask = label_mask[:, :max_text_len, :max_text_len, :]

        batch_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
            "label_token_idx": label_token_idx,
            "label_mask": label_mask,
            'text_len': text_len
        }

        if "span_labels" in batch[0].keys():
            span_labels = torch.zeros((batch_size,
                                       batch_max_length,
                                       batch_max_length,
                                       batch_max_labels))
            for i, encoded in enumerate(batch):
                input_len, _, sample_num_labels = encoded["span_labels"].shape
                span_labels[i, :input_len, :input_len, :sample_num_labels] = encoded["span_labels"]
            batch_data["span_labels"] = span_labels[:, :max_text_len, :max_text_len, :]

        return batch_data

    @staticmethod
    def collate_expand(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate a batch data and expand to full attention
        batch = [ins1_dict, ins2_dict, ..., insN_dict]
        batch_data = {"sentence":[ins1_sentence, ins2_sentence...],
        "input_ids":[ins1_input_ids, ins2_input_ids...], ...}
        """
        mask_atten_batch = ItemEncoder.collate(batch)
        full_atten_batch = ItemEncoder.collate(batch)
        # 对full_atten_batch进行改造
        atten_mask = full_atten_batch['attention_mask']
        b, _, _ = atten_mask.size()
        for i in range(b):
            ent_ranges, rel_ranges = batch[i]['ent_ranges'], batch[i]['rel_ranges']
            text_len = ent_ranges[0][0] # text长度

            if not rel_ranges:
                assert len(ent_ranges) == 1, 'ent_ranges:%s' % ent_ranges
                s, e = ent_ranges[0]
                atten_mask[i, : text_len, s: e] = 1
            else:
                assert len(rel_ranges) == 1 and len(ent_ranges) <= 2, \
                    'ent_ranges:%s, rel_ranges:%s' % (ent_ranges, rel_ranges)
                s, e = rel_ranges[0]
                atten_mask[i, : text_len, s: e] = 1
        full_atten_batch['attention_mask'] = atten_mask
        collate_batch = {}
        for key, value in mask_atten_batch.items():
            collate_batch[key] = torch.cat((value, full_atten_batch[key]), 0)
        return collate_batch
