from pathlib import Path
from typing import Union, Generator

from ...utils.json_utils import load_json_list, load_json
from .consts import (
    Label2Token,
    LabeledSample,
    UnlabeledSample,
    Label2Id,
    RawSample
)


Path_ = Union[str, Path]


class DataReaderClf:

    @classmethod
    def read_labeled_sample(cls, path: Path_, label2token: Label2Token) -> Generator[LabeledSample, None, None]:
        for raw_sample in cls.__read_raw_clf_sample(path):
            label_key = str(raw_sample.label)
            label = label2token[label_key].label
            label_id = label2token[label_key].label_id
            label_id_clf = label2token[label_key].label_id_clf
            labeled_sample = LabeledSample(
                text=raw_sample.content,
                id=raw_sample.id,
                label=label,
                label_id=label_id,
                label_id_clf=label_id_clf,
                soft_label=raw_sample.probs
            )
            yield labeled_sample

    @classmethod
    def read_unlabeled_sample(cls, path: Path_) -> Generator[UnlabeledSample, None, None]:
        for raw_sample in cls.__read_raw_clf_sample(path):
            yield UnlabeledSample(raw_sample.content, raw_sample.id)

    @classmethod
    def read_label2id(cls, path: Path_) -> Label2Id:
        return load_json(path, Label2Id)

    @classmethod
    def __read_raw_clf_sample(cls, path: Path_) -> Generator[RawSample, None, None]:
        return load_json_list(path, RawSample)
