from typing import Dict, Sequence, Protocol, List, TypeVar, Callable
import random

class LabeledSampleProto(Protocol):
    label: str
    
LabeledSample = TypeVar("LabeledSample", bound=LabeledSampleProto)

class SoftLabeledSampleProto(Protocol):
    soft_label: List[float]
    
SoftLabeledSample = TypeVar("SoftLabeledSample", bound=SoftLabeledSampleProto)

def balanced_sampling_by_label(labeled_sample_list: Sequence[LabeledSample], sample_num: int, minimum_num_per_label: int) -> Sequence[LabeledSample]:
    """按标签平衡抽样

    Args:
        labeled_sample_list (Sequence[LabeledSampleProto]): 抽样的sample列表，sample需有label字段
        sample_num (int): 总抽样数量
        minimum_num_per_label (int): 每个类别最小样本量
    """
    if len(labeled_sample_list) == 0:
        return []
    sample_list_by_label: Dict[str, List[LabeledSample]] = {}
    for sample in labeled_sample_list:
        if sample.label not in sample_list_by_label.keys():
            sample_list_by_label[sample.label] = [sample]
        else:
            sample_list_by_label[sample.label].append(sample)
    num_per_label = int(sample_num / len(sample_list_by_label))
    num_per_label = max(num_per_label, minimum_num_per_label)
    res: List[LabeledSample] = []
    for sample_list in sample_list_by_label.values():
        random.shuffle(sample_list)
        res += sample_list[:num_per_label]
    return res

def get_prob_threshold(label_classes: int, use_unlabel: bool):
    '''
    description: 根据类别确定筛选样本的置信度
                label_classes(int): 样本的个数
    return {prob_threshold(float): 置信度阈值}
    '''    
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

def filter_data_by_confidence(sample_list: Sequence[SoftLabeledSample], label_classes: int, use_unlabel: bool=True)-> Sequence[SoftLabeledSample]:
    '''
    description: 根据不同的类别和置信度过滤数据
                sample_list: Sequence[SoftLabeledSample]: 样本
                label_classes: int: 类别数目
    return {*}
    '''
    if len(sample_list) == 0:
        return []
    prob_threshold = get_prob_threshold(label_classes, use_unlabel)
    filter_fx: Callable[[SoftLabeledSample], bool] = lambda sample: len(sample.soft_label) > 0 and max(sample.soft_label) > prob_threshold
    filtered_sample_list = [sample for sample in sample_list if filter_fx(sample)]
    return filtered_sample_list