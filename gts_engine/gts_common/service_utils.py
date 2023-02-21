"""Copyright 2022 The International Digital Economy Academy (IDEA). CCNL team.
All rights reserved. Licensed under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at.

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@File    :   api_utils.py
@Time    :   2022/10/31 18:30
@Author  :   Kunhao Pan
@Version :   1.0
@Contact :   pankunhao@idea.edu.cn
@License :   (C)Copyright 2022-2023, CCNL-IDEA
"""

import json
import os
from typing import List, Optional, Tuple, Union

from gts_common.logs_utils import Logger
from pydantic import BaseModel, conint, conlist, constr

logger = Logger().get_log()


def list_task(task_dir):
    if not os.path.exists(task_dir):
        return []
    tasks = os.listdir(task_dir)
    return tasks


def is_task_valid(task_dir, task_id):
    tasks = list_task(task_dir)
    tasks = set(tasks)
    return (task_id in tasks)


def is_data_format_valid(data_path, data_type):
    logger.info(format(data_path))
    if not os.path.exists(data_path):
        return False
    valid = True
    with open(data_path, encoding='utf8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except BaseException:
                valid = False
                break
            if data_type == 'train' or data_type == 'dev':
                if "content" not in data or "label" not in data:
                    valid = False
                    break
            if data_type == 'test' or data_type == "unlabeled":
                if "content" not in data:
                    valid = False
                    break
            if data_type == 'label':
                if "labels" not in data:
                    valid = False
    return valid


class DataFormatChecker:
    """数据格式校验."""

    def check_from_path(self, task_type: str, data_type: str,
                        data_path: str) -> Tuple[bool, str]:
        """check.

        Args:
            task_type (str): task type
            data_type (str): data type
            data_path (str): data path

        Returns:
            Tuple[bool, str]: check result
        """
        if not os.path.exists(data_path):
            return False, f"{data_path}路径不存在"

        data = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except BaseException:
                    return False, f"行【{line}】非json格式"
                data.append(item)
        return self.check_data(task_type, data_type, data)

    def check_data(self, task_type: str, data_type: str,
                   data: List[dict]) -> Tuple[bool, str]:
        """check.

        Args:
            task_type (str): task_type
            data_type (str): data_type
            data (List[dict]): data

        Returns:
            Tuple[bool, str]: check result
        """
        if task_type == "classification":
            return self._check_classification_data(data_type, data)
        elif task_type == "similarity":
            return self._check_similarity_data(data_type, data)
        elif task_type == "nli":
            return self._check_nli_data(data_type, data)
        elif task_type == "ie":
            return self._check_ie_data(data_type, data)
        elif task_type == "summary":
            return self._check_summary_data(data_type, data)
        else:
            raise ValueError(
                f"task_type `{task_type}` format checking is not supported.")

    def _check_classification_data(self, data_type: str,
                                   data: List[dict]) -> Tuple[bool, str]:
        for item in data:
            if data_type in {"train", "dev"}:
                if "content" not in item:
                    return False, f"样本【{item}】格式错误：缺少`content`字段"
                if "label" not in item:
                    return False, f"样本【{item}】格式错误：缺少`label`字段"
            if data_type in {"test", "unlabeled"}:
                if "content" not in item:
                    return False, f"样本【{item}】格式错误：缺少`content`字段"
            if data_type in {"label"}:
                if "labels" not in item:
                    return False, "标签描述格式错误：缺少`labels`字段"

        return True, "OK"

    def _check_similarity_data(self, data_type: str,
                               data: List[dict]) -> Tuple[bool, str]:
        # TODO: 补充
        return True, "OK"

    def _check_nli_data(self, data_type: str,
                        data: List[dict]) -> Tuple[bool, str]:
        # TODO: 补充
        return True, "OK"

    def _check_ie_data(self, data_type: str,
                       data: List[dict]) -> Tuple[bool, str]:

        class Entity(BaseModel):
            """entity."""
            entity_text: constr(min_length=1)
            entity_type: constr(min_length=1)
            entity_index: Tuple[conint(gt=0), conint(gt=0)]

        class SPO(BaseModel):
            """spo."""
            predicate: constr(min_length=1)
            subject: Entity
            object: Entity

        class IESample(BaseModel):
            """sample."""

            regex = "(实体识别|关系抽取)"
            task: constr(regex=regex)
            text: constr(min_length=1)
            entity_list: Optional[List[Entity]]
            spo_list: Optional[List[SPO]]
            choice: conlist(Union[constr(min_length=1),
                                  Tuple[constr(min_length=1),
                                        constr(min_length=1),
                                        constr(min_length=1)]],
                            min_items=1)

        for item in data:
            try:
                IESample.parse_obj(item)
            except Exception as e:
                return False, f"样本【{item}】格式错误：\n{e}"

            # if data_type in {"train", "val", "dev"}:
            #     pass

        return True, "OK"

    def _check_summary_data(self, data_type: str,
                            data: List[dict]) -> Tuple[bool, str]:
        for item in data:
            if "text" not in item:
                return False, f"样本【{item}】格式错误：缺少`text`字段"
            if "summary" not in item:
                return False, f"样本【{item}】格式错误：缺少`summary`字段"

        return True, "OK"
