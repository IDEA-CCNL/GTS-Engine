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

from typing import List, Tuple


def eval_entity_f1(test_data: List[dict], pred_data: List[dict]) -> Tuple[float, tuple, tuple]:
    """ 获取实体抽取f1

    Args:
        test_data (List[dict]): test data
        pred_data (List[dict]): pred data

    Returns:
        Tuple[float, tuple, tuple]: (f1, recall, precise)
    """
    corr = 0
    y_true = 0
    y_pred = 0
    assert len(test_data) == len(pred_data)
    for test_sample, pred_sample in zip(test_data, pred_data):
        assert test_sample['text'] == pred_sample['text']

        y_true_set = {(e["entity_type"], e["entity_text"].strip()) for e in test_sample["entity_list"]}
        y_true += len(y_true_set)

        y_pred_set = {(e["entity_type"], e["entity_text"].strip()) for e in pred_sample["entity_list"]}
        y_pred += len(y_pred_set)

        corr += len(y_pred_set & y_true_set)


    if y_pred <= 0:
        precise = 0.
    else:
        precise = corr / y_pred

    if y_true <= 0:
        recall = 0.
    else:
        recall = corr / y_true
    if precise + recall <= 0:
        f1 = 0.
    else:
        f1 = 2 * precise * recall / (precise + recall)

    return f1, (corr, y_true, recall), (corr, y_pred, precise)


def eval_relation_f1(test_data: List[dict], pred_data: List[dict]) -> Tuple[float, tuple, tuple]:
    """ 获取关系抽取f1

    Args:
        test_data (List[dict]): test data
        pred_data (List[dict]): pred data

    Returns:
        Tuple[float, tuple, tuple]: (f1, recall, precise)
    """
    corr = 0
    y_true = 0
    y_pred = 0

    for test_sample, pred_sample in zip(test_data, pred_data):

        y_true_set = {
            (spo["predicate"], spo["subject"]["entity_text"].strip(), spo["object"]["entity_text"].strip())
            for spo in test_sample["spo_list"]
        }
        y_true += len(y_true_set)

        y_pred_set = {
            (spo["predicate"], spo["subject"]["entity_text"].strip(), spo["object"]["entity_text"].strip())
            for spo in pred_sample["spo_list"]
        }
        y_pred += len(y_pred_set)

        corr += len(y_pred_set & y_true_set)

    if y_pred <= 0:
        precise = 0.
    else:
        precise = corr / y_pred

    if y_true <= 0:
        recall = 0.
    else:
        recall = corr / y_true

    if precise + recall <= 0:
        f1 = 0.
    else:
        f1 = 2 * precise * recall / (precise + recall)

    return f1, (corr, y_true, recall), (corr, y_pred, precise)


def eval_entity_relation_f1(test_data: List[dict], pred_data: List[dict]) -> Tuple[float, tuple, tuple]:
    """ 获取实体关系抽取f1

    Args:
        test_data (List[dict]): test data
        pred_data (List[dict]): pred data

    Returns:
        Tuple[float, tuple, tuple]: (f1, recall, precise)
    """
    corr = 0
    y_true = 0
    y_pred = 0

    for test_sample, pred_sample in zip(test_data, pred_data):

        y_true_set = {
            (spo["predicate"], spo["subject"]["entity_text"].strip(), spo["subject"]["entity_type"],
            spo["object"]["entity_text"].strip(), spo["object"]["entity_type"])
            for spo in test_sample["spo_list"]
        }
        y_true += len(y_true_set)

        y_pred_set = {
            (spo["predicate"], spo["subject"]["entity_text"].strip(), spo["subject"]["entity_type"],
            spo["object"]["entity_text"].strip(), spo["object"]["entity_type"])
            for spo in pred_sample["spo_list"]
        }
        y_pred += len(y_pred_set)

        corr += len(y_pred_set & y_true_set)

    if y_pred <= 0:
        precise = 0.
    else:
        precise = corr / y_pred

    if y_true <= 0:
        recall = 0.
    else:
        recall = corr / y_true

    if precise + recall <= 0:
        f1 = 0.
    else:
        f1 = 2 * precise * recall / (precise + recall)

    return f1, (corr, y_true, recall), (corr, y_pred, precise)
