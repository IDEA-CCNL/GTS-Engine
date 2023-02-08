from collections import defaultdict
from dataclasses import asdict
from typing import Protocol, Sequence

from torch.utils.data import Dataset

from .consts import PreTrainedTokenizer
from .label import StdLabel


class RawSampleProto(Protocol):
    text: str


class DatasetArgsProto(Protocol):
    inference_prompt: str
    prefix_prompt: str


class BaseDatasetClf(Dataset):

    def __init__(self, sample_list: Sequence[RawSampleProto],
                 tokenizer: PreTrainedTokenizer, label: StdLabel):
        self._tokenizer = tokenizer
        self._label = label
        self._pre_encoded_sample_list = [
            self._encode_before_iter(sample, idx)
            for idx, sample in enumerate(sample_list)
        ]
        self.sample_list = sample_list
        self._classwise_indices = defaultdict(list)
        for id in range(len(sample_list)):
            item = sample_list[id]
            if hasattr(item, "label_id_clf"):
                self._classwise_indices[item.label_id_clf].append(
                    id)  # type: ignore

    def __len__(self):
        return len(self._pre_encoded_sample_list)

    def __getitem__(self, index):
        encoded_sample = self._encode_on_iter(
            self._pre_encoded_sample_list[index], index)
        return asdict(encoded_sample)

    def __getclass__(self, index):
        return self.sample_list[index].label_id_clf  # type: ignore

    def _encode_before_iter(self, sample: RawSampleProto, idx: int):
        """迭代前编码."""
        return sample

    def _encode_on_iter(self, sample, idx: int):
        """迭代中编码.

        用于有随机成分的编码，使每次迭代时编码不同
        """
        return sample
