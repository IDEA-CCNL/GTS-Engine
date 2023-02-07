from abc import ABCMeta, abstractmethod
from typing import Dict, Union

from pydantic import FilePath

from .consts import Label, LabelParseOutput, LabelToken
from .data_reader_clf import DataReaderClf


class LabelBase(metaclass=ABCMeta):

    def __init__(self, label2id_path: Union[str, FilePath]) -> None:
        self._mask_token = '[MASK]'
        self._label2id_path = label2id_path
        self._label_des2tag = DataReaderClf.read_label2id(label2id_path)
        self.parse_label()

    @abstractmethod
    def parse_label(self) -> LabelParseOutput:
        pass

    @property
    def mask_token(self):
        return self._mask_token


class StdLabel(LabelBase):
    """通用Label."""

    def parse_label(self) -> LabelParseOutput:
        label2token: Dict[str, LabelToken] = {}
        id2label: Dict[int, Label] = {}
        for i, item in enumerate(self._label_des2tag.items()):
            key, val = item
            if val.label_desc_zh is None:
                raise Exception("label2id has no field \"label_desc_zh\"")
            label = val.label_desc_zh
            id = val.id
            label2token[key] = LabelToken(label, i, id, key)
            id2label[id] = Label(label, key)

        label_ids = [ele for ele in label2token.values()]
        label_ids.sort()
        self.label2token = label2token
        self.id2label = id2label
        self.label_ids = label_ids
        return LabelParseOutput(label2token, id2label, label_ids)
