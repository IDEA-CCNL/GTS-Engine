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

# from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
from transformers import PreTrainedTokenizer

from .item_encoder import entity_based_tokenize, get_entity_indices
from .dataset_utils import get_choice
from ...arguments.ie import TrainingArgumentsIEStd


class ItemDecoder(object):
    """ Decoder

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer
        args (TrainingArgumentsIEStd): arguments
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 args: TrainingArgumentsIEStd) -> None:
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.threshold_entity = args.threshold_ent
        self.threshold_rel = args.threshold_rel
        self.entity_multi_label = args.entity_multi_label
        self.relation_multi_label = args.relation_multi_label

    def extract_entity_index(self,
                             entity_logits: np.ndarray,
                             ) -> List[Tuple[int, int]]:
        """ extract entity index

        Args:
            entity_logits (np.ndarray): entity_logits

        Returns:
            List[Tuple[int, int]]: result
        """

        l, _, d = entity_logits.shape
        result = []
        for i in range(l):
            for j in range(i, l):
                if self.entity_multi_label:
                    for k in range(d):
                        entity_score = float(entity_logits[i, j, k])
                        if entity_score > self.threshold_entity:
                            result.append((i, j, k, entity_score))

                else:
                    k = np.argmax(entity_logits[i, j])
                    entity_score = float(entity_logits[i, j, k])
                    if entity_score > self.threshold_entity:
                        result.append((i, j, k, entity_score))

        return result

    @staticmethod
    def extract_entity(text: str,
                       entity_idx: List[int],
                       entity_type: str,
                       entity_score: float,
                       text_start_id: int,
                       offset_mapping: List[List[int]]) -> dict:
        """ extract entity

        Args:
            text (str): text
            entity_idx (List[int]): entity indices
            entity_type (str): entity type
            entity_score (float): entity score
            text_start_id (int): text_start_id
            offset_mapping (List[List[int]]): offset mapping

        Returns:
            dict: entity
        """
        entity_start, entity_end = entity_idx[0] - text_start_id, entity_idx[1] - text_start_id

        start_split = offset_mapping[entity_start] if 0 <= entity_start < len(offset_mapping) else []
        end_split = offset_mapping[entity_end] if 0 <= entity_end < len(offset_mapping) else []

        if not start_split or not end_split:
            return None

        start_idx, end_idx = start_split[0], end_split[-1]
        entity_text = text[start_idx: end_idx]

        if not entity_text:
            return None

        entity = {
            "entity_text": entity_text,
            "entity_type": entity_type,
            "score": entity_score,
            "entity_index": [start_idx, end_idx]
        }

        return entity

    def decode_ner(self,
                   text: str,
                   choice: List[str],
                   sample_span_logits: np.ndarray,
                   offset_mapping: List[List[int]]
                  ) -> List[dict]:
        """ NER decode

        Args:
            text (str): text
            choice (List[str]): choice
            sample_span_logits (np.ndarray): sample span_logits
            offset_mapping (List[List[int]]): offset mapping


        Returns:
            List[dict]: decoded entity list
        """
        entity_list = []

        entity_idx_list = self.extract_entity_index(sample_span_logits)

        for entity_start, entity_end, entity_type_idx, entity_score in entity_idx_list:

            entity = self.extract_entity(text,
                                         [entity_start, entity_end],
                                         choice[entity_type_idx],
                                         entity_score,
                                         text_start_id=1,
                                         offset_mapping=offset_mapping)

            if entity is None:
                continue

            if entity not in entity_list:
                entity_list.append(entity)

        return entity_list

    def decode_spo(self,
                   text: str,
                   choice: List[List[str]],
                   sample_span_logits: np.ndarray,
                   offset_mapping: List[List[int]]) -> tuple:
        """ SPO decode

        Args:
            text (str): text
            choice (List[List[str]]): choice
            sample_span_logits (np.ndarray): sample span_logits
            offset_mapping (List[List[int]): offset mapping

        Returns:
            List[dict]: decoded spo list
            List[dict]: decoded entity list
        """
        spo_list = []
        entity_list = []

        choice_ent, choice_rel, choice_head, choice_tail, entity2rel = get_choice(choice)

        entity_logits = sample_span_logits[:, :, : len(choice_ent)] # (seq_len, seq_len, num_entity)
        relation_logits = sample_span_logits[:, :, len(choice_ent): ] # (seq_len, seq_len, num_relation)

        entity_idx_list = self.extract_entity_index(entity_logits)

        head_list = []
        tail_list = []
        for entity_start, entity_end, entity_type_idx, entity_score in entity_idx_list:

            entity_type = choice_ent[entity_type_idx]

            entity = self.extract_entity(text,
                                         [entity_start, entity_end],
                                         entity_type,
                                         entity_score,
                                         text_start_id=1,
                                         offset_mapping=offset_mapping)

            if entity is None:
                continue

            if entity_type in choice_head:
                head_list.append((entity_start, entity_end, entity_type, entity))
            if entity_type in choice_tail:
                tail_list.append((entity_start, entity_end, entity_type, entity))

        for head_start, head_end, subject_type, subject_dict in head_list:
            for tail_start, tail_end, object_type, object_dict in tail_list:

                if subject_dict == object_dict:
                    continue

                if (subject_type, object_type) not in entity2rel.keys():
                    continue

                relation_candidates = entity2rel[subject_type, object_type]
                rel_idx = [choice_rel.index(r) for r in relation_candidates]

                so_rel_logits = relation_logits[:, :, rel_idx]

                if self.relation_multi_label:
                    for idx, predicate in enumerate(relation_candidates):
                        rel_score = so_rel_logits[head_start, tail_start, idx] + \
                                    so_rel_logits[head_end, tail_end, idx]
                        predicate_score = float(rel_score / 2)

                        if predicate_score <= self.threshold_rel:
                            continue

                        if subject_dict not in entity_list:
                            entity_list.append(subject_dict)
                        if object_dict not in entity_list:
                            entity_list.append(object_dict)

                        spo = {
                            "predicate": predicate,
                            "score": predicate_score,
                            "subject": subject_dict,
                            "object": object_dict,
                        }

                        if spo not in spo_list:
                            spo_list.append(spo)

                else:

                    hh_idx = np.argmax(so_rel_logits[head_start, head_end])
                    tt_idx = np.argmax(so_rel_logits[tail_start, tail_end])
                    hh_score = so_rel_logits[head_start, tail_start, hh_idx] + so_rel_logits[head_end, tail_end, hh_idx]
                    tt_score = so_rel_logits[head_start, tail_start, tt_idx] + so_rel_logits[head_end, tail_end, tt_idx]

                    predicate = relation_candidates[hh_idx] if hh_score > tt_score else relation_candidates[tt_idx]

                    predicate_score = float(max(hh_score, tt_score) / 2)

                    if predicate_score <= self.threshold_rel:
                        continue

                    if subject_dict not in entity_list:
                        entity_list.append(subject_dict)
                    if object_dict not in entity_list:
                        entity_list.append(object_dict)

                    spo = {
                        "predicate": predicate,
                        "score": predicate_score,
                        "subject": subject_dict,
                        "object": object_dict,
                    }

                    if spo not in spo_list:
                        spo_list.append(spo)

        return spo_list, entity_list

    def decode(self,
               item: Dict,
               span_logits: np.ndarray,
               label_mask:  np.ndarray,
               ):
        """ decode

        Args:
            task (str): task name
            choice (list): choice
            text (str): text
            span_logits (np.ndarray): sample span_logits
            label_mask (np.ndarray): label_mask

        Raises:
            NotImplementedError: raised if task name is not supported

        Returns:
            List[dict]: decoded entity list
            List[dict]: decoded spo list
        """
        task, choice, text = item["task"], item["choice"], item["text"]
        entity_indices = get_entity_indices(item.get("entity_list", []), item.get("spo_list", []))
        _, offset_mapping = entity_based_tokenize(text, self.tokenizer, entity_indices,
                                                  return_offsets_mapping=True)

        assert span_logits.shape == label_mask.shape

        span_logits = span_logits + (label_mask - 1) * 100000

        spo_list = []
        entity_list = []

        if task in {"实体识别", "抽取任务"}:
            entity_list = self.decode_ner(text,
                                          choice,
                                          span_logits,
                                          offset_mapping)

        elif task in {"关系抽取"}:
            spo_list, entity_list = self.decode_spo(text,
                                                    choice,
                                                    span_logits,
                                                    offset_mapping)

        else:
            raise NotImplementedError

        return entity_list, spo_list
