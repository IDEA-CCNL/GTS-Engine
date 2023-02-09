"""抽样、样本筛选相关工具集."""
import random
from typing import Callable, Dict, List, Protocol, Sequence, TypeVar


class LabeledSampleProto(Protocol):
    """带标签数据类型协议.

    包含可读的label属性
    """

    @property
    def label(self) -> str:
        ...


LabeledSampleType = TypeVar("LabeledSampleType", bound=LabeledSampleProto)


class SoftLabeledSampleProto(Protocol):
    """带软标签数据类型协议.

    包含可读的soft_label属性
    """
    soft_label: List[float]


SoftLabeledSampleType = TypeVar("SoftLabeledSampleType",
                                bound=SoftLabeledSampleProto)


def balanced_sampling_by_label(
        labeled_sample_list: Sequence[LabeledSampleType], sample_num: int,
        minimum_num_per_label: int) -> List[LabeledSampleType]:
    """按标签平衡抽样.

    Args:
        labeled_sample_list (Sequence[LabeledSampleProto]):
            抽样的sample列表，sample需有label字段
        sample_num (int):
            总抽样数量
        minimum_num_per_label (int):
            每个类别最小样本量
    """
    if len(labeled_sample_list) == 0:
        return []
    sample_list_by_label: Dict[str, List[LabeledSampleType]] = {}
    for sample in labeled_sample_list:
        if sample.label not in sample_list_by_label.keys():
            sample_list_by_label[sample.label] = [sample]
        else:
            sample_list_by_label[sample.label].append(sample)
    num_per_label = int(sample_num / len(sample_list_by_label))
    num_per_label = max(num_per_label, minimum_num_per_label)
    res: List[LabeledSampleType] = []
    for sample_list in sample_list_by_label.values():
        random.shuffle(sample_list)
        res += sample_list[:num_per_label]
    return res


def get_prob_threshold(label_classes: int, use_unlabel: bool) -> float:
    """根据类别确定筛选样本的置信度.

    Args:
        label_classes (int):
            样本的个数
        use_unlabel (bool):
            TODO (JiangYuzhen) 不太清楚什么意思，待补全

    Returns:
        float: 样本置信度阈值
    """
    if label_classes <= 2:
        prob_threshold = 0.95
    elif label_classes <= 10:
        prob_threshold = 0.9
    elif label_classes <= 20:
        prob_threshold = 0.9
    elif label_classes <= 40:
        prob_threshold = 0.75
    else:
        prob_threshold = 0.5

    if not use_unlabel:
        prob_threshold = prob_threshold - 0.1

    return prob_threshold


def filter_data_by_confidence(
        sample_list: Sequence[SoftLabeledSampleType],
        label_classes: int,
        use_unlabel: bool = True) -> List[SoftLabeledSampleType]:
    """根据不同的类别和置信度过滤数据.

    Args:
        sample_list (Sequence[SoftLabeledSampleType]):
            样本列表
        label_classes (int):
            类别数目
        use_unlabel (bool, optional):
            TODO (JiangYuzhen) 不太清楚什么意思，待补全

    Returns:
        List[SoftLabeledSampleType]: 过滤后的样本列表
    """
    if len(sample_list) == 0:
        return []
    prob_threshold = get_prob_threshold(label_classes, use_unlabel)
    filter_fx: Callable[[SoftLabeledSampleType], bool] = (lambda sample: len(
        sample.soft_label) > 0 and max(sample.soft_label) > prob_threshold)
    filtered_sample_list = [
        sample for sample in sample_list if filter_fx(sample)
    ]
    return filtered_sample_list
