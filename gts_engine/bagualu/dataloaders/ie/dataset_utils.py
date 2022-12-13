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

import collections
from typing import List, Dict, Tuple


def get_choice(spo_choice: list) -> tuple:
    """ 把关系schema中的关系、实体获取出来

    Args:
        spo_choice (list): 关系schema

    Returns:
        tuple:
            choice_ent (list)
            choice_rel (list)
            choice_head (list)
            choice_tail (list)
            entity2rel (dict)
    """
    choice_head = []
    choice_tail = []
    choice_ent = []
    choice_rel = []
    entity2rel = collections.defaultdict(list) # "subject|object" -> [relation]

    for head, rel, tail in spo_choice:

        if head not in choice_head:
            choice_head.append(head)
        if tail not in choice_tail:
            choice_tail.append(tail)

        if head not in choice_ent:
            choice_ent.append(head)
        if tail not in choice_ent:
            choice_ent.append(tail)

        if rel not in choice_rel:
            choice_rel.append(rel)

        entity2rel[head, tail].append(rel)

    return choice_ent, choice_rel, choice_head, choice_tail, entity2rel


def check_data(data: List) -> None:
    """ 检查数据是否格式合法

    Args:
        data (List): 数据
    """

    def check_entity(entity: Dict,
                     choice_entity: List[str],
                     text: str) -> None:
        """ 检查实体格式

        Args:
            entity (Dict): 实体
            choice_entity (List[str]): 实体类型
            text (str): 文本
        """
        assert "entity_text" in entity
        assert "entity_type" in entity
        assert "entity_index" in entity
        assert text[entity["entity_index"][0]: entity["entity_index"][1]] == entity["entity_text"]
        assert entity["entity_type"] in choice_entity

    def check_spo(spo: Dict,
                  choice_entity: List[str],
                  entity2relation: Dict[Tuple[str, str], str],
                  text: str) -> None:
        """ 检查SPO格式

        Args:
            spo (Dict): SPO
            choice_entity (List[str]): 实体类型
            entity2relation (Dict[Tuple[str, str], str]): 实体-关系映射
            text (str): 文本
        """
        assert "predicate" in spo
        assert "subject" in spo
        assert "object" in spo
        check_entity(spo["subject"], choice_entity, text)
        check_entity(spo["object"], choice_entity, text)
        assert spo["predicate"] in entity2relation[spo["subject"]["entity_type"],
                                                   spo["object"]["entity_type"]]

    for item in data:

        assert "task" in item and isinstance(item["task"], str)
        assert "text" in item and isinstance(item["text"], str)
        assert "choice" in item

        if item["task"] == "实体识别":
            assert "entity_list" in item
            choice_entity = item["choice"]
            for entity in item["entity_list"]:
                check_entity(entity, choice_entity, item["text"])

        elif item["task"] == "关系抽取":
            assert "spo_list" in item
            choice_entity = get_choice(item["choice"])[0]
            entity2relation = get_choice(item["choice"])[-1]
            for spo in item["spo_list"]:
                check_spo(spo, choice_entity, entity2relation, item["text"])

        else:
            raise ValueError(f"task type `{item['task']}` is not supported yet.")


def data_segment(ori_data: List[dict],
                 cut_text_len: int = 400,
                 cut_text_stride: int = 200) -> List[dict]:
    """ 数据按照文本切割

    Args:
        ori_data (List[dict]): 原始数据
        cut_text_len (int, optional): 切割文本窗口大小. Defaults to 400.
        cut_text_stride (int, optional): 切割文本步长. Defaults to 200.

    Returns:
        List[dict]: 切割后的文本
    """
    data = []

    for item_id, item in enumerate(ori_data):

        task = item["task"]
        text = item["text"]
        entity_list = item.get("entity_list", [])
        spo_list = item.get("spo_list", [])
        choice = item["choice"]

        win_size = min(len(text), cut_text_len) # 实际窗口大小

        for win_end in range(win_size, len(text) + 1, cut_text_stride):

            win_start = win_end - win_size
            text_win = text[win_start: win_end] # 窗口文本

            # 处理窗口包含的实体
            entity_list_win = []
            for entity in entity_list:

                entity_start, entity_end = entity["entity_index"]
                entity_text, entity_type = entity["entity_text"], entity["entity_type"]

                if entity_start >= win_start and entity_end <= win_end: # 包含的实体
                    entity_index = [entity_start - win_start, entity_end - win_start] # 窗口中的相对索引
                    assert text_win[entity_index[0]: entity_index[1]] == entity["entity_text"]
                    entity_list_win.append({
                        "entity_text": entity_text,
                        "entity_type": entity_type,
                        "entity_index": entity_index
                    })

            # 处理窗口包含的关系
            spo_list_win = []
            for spo in spo_list:

                predicate = spo["predicate"]
                subj_start, subj_end = spo["subject"]["entity_index"]
                subj_text, subj_type = spo["subject"]["entity_text"], spo["subject"]["entity_type"]
                obj_start, obj_end = spo["object"]["entity_index"]
                obj_text, obj_type = spo["object"]["entity_text"], spo["object"]["entity_type"]

                if subj_start >= win_start and subj_end <= win_end and \
                        obj_start >= win_start and obj_end <= win_end: # 包含的关系
                    subj_index = [subj_start - win_start, subj_end - win_start]
                    obj_index = [obj_start - win_start, obj_end - win_start]
                    assert text_win[subj_index[0]: subj_index[1]] == subj_text
                    assert text_win[obj_index[0]: obj_index[1]] == obj_text
                    spo_list_win.append({
                        "predicate": predicate,
                        "subject": {
                            "entity_text": subj_text,
                            "entity_type": subj_type,
                            "entity_index": subj_index,
                        },
                        "object": {
                            "entity_text": obj_text,
                            "entity_type": obj_type,
                            "entity_index": obj_index,
                        }
                    })

            data.append({
                "id": item_id,
                "task": task,
                "text": text_win,
                "entity_list": entity_list_win,
                "spo_list": spo_list_win,
                "choice": choice,
                "bias": win_start,
                "full_text": text,
                "full_entity_list": entity_list,
                "full_spo_list": spo_list,
            })

    return data


def data_segment_restore(ori_data: List[dict]) -> List[dict]:
    """ 切割后的数据进行预测后，对预测结果进行合并

    Args:
        ori_data (List[dict]): 原始数据

    Returns:
        List[dict]: 合并后数据
    """
    ori_data.sort(key=lambda x: (x["id"], x["bias"]))  # 保证相同ID的样本连续，且合并后样本顺序与原始数据相同

    data = []
    for item in ori_data:
        task = item["task"]
        text = item["full_text"]
        entity_list_win = item["entity_list"]
        spo_list_win = item["spo_list"]
        choice = item["choice"]
        bias = item["bias"]

        if bias == 0:
            new_item = {
                "task": task,
                "text": text,
                "entity_list": [],
                "spo_list": [],
                "choice": choice,
            }
            data.append(new_item)
        else:
            new_item = data[-1]
            assert new_item["text"] == text

        # 合并实体列表
        for entity in entity_list_win:
            entity_text, entity_type, score = entity["entity_text"], \
                                              entity["entity_type"], entity["score"]
            entity_start, entity_end = entity["entity_index"][0] + bias, \
                                       entity["entity_index"][1] + bias
            assert entity_text == text[entity_start: entity_end]

            # 判断该实体是否已经被前一个窗口包含，若包含则更新score，不包含则加入
            is_included = False
            entity_key = (entity_text, entity_type, entity_start, entity_end) # 判断实体被包含的key
            for pre_entity in new_item["entity_list"]:
                pre_entity_key =  (pre_entity["entity_text"], pre_entity["entity_type"],
                                   pre_entity["entity_index"][0], pre_entity["entity_index"][1])
                if entity_key == pre_entity_key:
                    pre_entity["score"] = max(pre_entity["score"], score) # score取最高
                    is_included = True
                    break

            if not is_included:
                new_item["entity_list"].append({
                    "entity_text": entity_text,
                    "entity_type": entity_type,
                    "entity_index": [entity_start, entity_end],
                    "score": score,
                })

        # 合并关系列表
        for spo in spo_list_win:
            predicate, subj, obj, score = spo["predicate"], spo["subject"], \
                                          spo["object"], spo["score"]
            subj_text, subj_type, subj_score = subj["entity_text"], \
                                               subj["entity_type"], subj["score"]
            subj_start, subj_end = subj["entity_index"][0] + bias, \
                                   subj["entity_index"][1] + bias
            obj_text, obj_type, obj_score = obj["entity_text"], obj["entity_type"], obj["score"]
            obj_start, obj_end = obj["entity_index"][0] + bias, obj["entity_index"][1] + bias
            assert subj_text == text[subj_start: subj_end]
            assert obj_text == text[obj_start: obj_end]

            # 判断该关系是否已经被前一个窗口包含，若包含则更新score，不包含则加入
            is_included = False
            spo_key = (predicate, subj_text, subj_type, subj_start, subj_end,
                       obj_text, obj_type, obj_start, obj_end)
            for pre_spo in new_item["spo_list"]:
                pre_subj, pre_obj = pre_spo["subject"], pre_spo["object"]
                pre_spo_key = (pre_spo["predicate"], pre_subj["entity_text"],
                               pre_subj["entity_type"], pre_subj["entity_index"][0],
                               pre_subj["entity_index"][1], pre_obj["entity_text"],
                               pre_obj["entity_type"], pre_obj["entity_index"][0],
                               pre_obj["entity_index"][1])
                if spo_key == pre_spo_key:
                    pre_spo["score"] = max(pre_spo["score"], score) # score取最高
                    is_included = True
                    break

            if not is_included:
                new_item["spo_list"].append({
                    "predicate": predicate,
                    "subject": {
                        "entity_text": subj_text,
                        "entity_type": subj_type,
                        "entity_index": [subj_start, subj_end],
                        "score": subj_score
                    },
                    "object": {
                        "entity_text": obj_text,
                        "entity_type": obj_type,
                        "entity_index": [obj_start, obj_end],
                        "score": obj_score
                    },
                    "score": score
                })

    return data
