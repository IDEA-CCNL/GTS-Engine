from ctypes import Union
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Union, Optional
from sklearn.metrics import classification_report, confusion_matrix

@dataclass
class GetConfusionMatrixOutput:
    confusion_matrix: List
    labels: List[str]

def get_confusion_matrix(y_true: List[int], y_pred: List[int], id2label: Dict[int, str]) -> GetConfusionMatrixOutput:
    label_ids = sorted(list(set(y_true + y_pred)))
    label_names = [id2label[i] for i in label_ids]
    return GetConfusionMatrixOutput(confusion_matrix(y_true=y_true, y_pred=y_pred, labels=label_ids).tolist(), label_names)

def get_classification_report(y_true: List[int], y_pred: List[int], id2label: Dict[int, str]) -> Dict[str, Any]:
    label_ids = sorted(list(set(y_true + y_pred)))
    label_names = [id2label[i] for i in label_ids]
    return classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        digits=2,
        zero_division=0,    # type: ignore
        output_dict=True
    ) 