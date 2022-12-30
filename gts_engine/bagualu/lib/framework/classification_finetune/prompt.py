from typing import Any, Dict, NamedTuple, Union, List
from abc import abstractmethod, ABCMeta
from pydantic import FilePath

from .consts import PromptLabelParseOutput, PromptToken, PromptLabel
from .data_reader_clf import DataReaderClf

class PromptBase(metaclass=ABCMeta):
    
    def __init__(self, prompt: str, label2id_path: Union[str, FilePath]) -> None:
        self._mask_id = 103
        self._mask_token = '[MASK]'
        self._prompt = prompt
        self._label2id_path = label2id_path
        self._label_des2tag = DataReaderClf.read_label2id(label2id_path)
        self.parse_label()
    
    @abstractmethod
    def parse_label(self) -> PromptLabelParseOutput:
        pass
    
    @property
    def prompt(self):
        return self._prompt
    
    @property
    def mask_token(self):
        return self._mask_token

class StdPrompt(PromptBase):
    """通用Prompt"""
    def parse_label(self) -> PromptLabelParseOutput:
        label2token: Dict[str, PromptToken] = {}
        id2label: Dict[int, PromptLabel] = {}
        for i, item in enumerate(self._label_des2tag.items()):
            key, val = item
            if val.label_desc_zh is None:
                raise Exception("label2id has no field \"label_desc_zh\"")
            label = val.label_desc_zh
            id = val.id
            label2token[key] = PromptToken(label, i, id, key)
            id2label[id] = PromptLabel(label, key)
            
        label_ids = [ele for ele in label2token.values()]
        label_ids.sort()
        self.label2token = label2token
        self.id2label = id2label
        self.label_ids = label_ids
        return PromptLabelParseOutput(label2token, id2label, label_ids)