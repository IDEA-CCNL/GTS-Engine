from dataclasses import dataclass
from typing import Any, Dict, List

from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class GetConfusionMatrixOutput:
    confusion_matrix: List
    labels: List[str]


def get_confusion_matrix(y_true: List[int], y_pred: List[int],
                         id2label: Dict[int, str]) -> GetConfusionMatrixOutput:
    """获取混淆矩阵.

    Args:
        y_true (List[int]): 真实标签
        y_pred (List[int]): 预测标签
        id2label (Dict[int, str]): 样本对应label2id

    Returns:
        GetConfusionMatrixOutput: 返回结构体，包含属性
            confusion_matrix(List): 混淆矩阵
            labels(List[str]): 混淆矩阵每一行对应的标签
    """
    label_ids = sorted(list(set(y_true + y_pred)))
    label_names = [id2label[i] for i in label_ids]
    return GetConfusionMatrixOutput(
        confusion_matrix(y_true=y_true, y_pred=y_pred,
                         labels=label_ids).tolist(), label_names)


def get_classification_report(y_true: List[int], y_pred: List[int],
                              id2label: Dict[int, str]) -> Dict[str, Any]:
    """获取分类任务结果报告.

    Args:
        y_true (List[int]): 真实标签
        y_pred (List[int]): 预测标签
        id2label (Dict[int, str]): 样本对应label2id

    Returns:
        Dict[str, Any]: sklearn classification_report
    """
    label_ids = sorted(list(set(y_true + y_pred)))
    label_names = [id2label[i] for i in label_ids]
    return classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        digits=2,
        zero_division=0,  # type: ignore
        # TODO (Jiang Yuzhen) 这里zero_division的参数类型应该是str，不知为什么是0，再检查一下
        output_dict=True)  # type: ignore
