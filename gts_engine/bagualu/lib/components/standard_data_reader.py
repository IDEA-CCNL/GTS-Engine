from typing import Optional, Generator, NewType, Dict, List, Union
from tqdm import tqdm
from dataclasses import dataclass, field
from pydantic import FilePath

from ..framework.classification_finetune.consts import (
    Label2Token, 
    LabeledSample, 
    UnlabeledSample, 
    Label2Id, 
    Label2IdValue, 
    RawSample
)
from ..utils.json import load_json, load_json_list

#############################################################################################
## data reader
#############################################################################################


class StdDataReader:
    
    @classmethod
    def load_labeled_sample(
        cls, 
        path: Union[str, FilePath], 
        label2token: Label2Token
    ) -> Generator[LabeledSample, None, None]:
        """加载数据，并做初步加工"""
        for raw_sample in cls.__load_raw_sample(path):
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
    def load_unlabeled_sample(cls, path: Union[str, FilePath]) -> Generator[UnlabeledSample, None, None]:
        for raw_sample in cls.__load_raw_sample(path):
            yield UnlabeledSample(raw_sample.content, raw_sample.id)
                
    @classmethod
    def load_label2id(cls, path: Union[str, FilePath]) -> Label2Id:
        """加载label2id文件，并检查字段"""
        raw_label2id = load_json(path)
        if not isinstance(raw_label2id, dict) or not isinstance(val := list(raw_label2id.values())[0], dict):
            raise Exception("label2id file must be in the format \"Dict[label, dict]\"")
        if not set(["id", "label_desc_zh"]).issubset(val.keys()):
            raise Exception("label2id file does not have right fields")
        return {key: Label2IdValue(**val) for key, val in raw_label2id.items()}
    
    @classmethod
    def __load_raw_sample(cls, path: Union[str, FilePath]) -> Generator[RawSample, None, None]:
        """从数据文件中加载数据，并检查字段，不做加工"""
        check_sample = next(load_json_list(path))
        if not isinstance(check_sample, dict) or not {"content", "id"}.issubset(check_sample.keys()):
            raise Exception("sample file is not in the right format")
        # 构造RawSample的时候会自动检查必要字段，但为了异常的可读性，添加手动检查，只检查第一条数据
        
        for raw_sample_dict in load_json_list(path):
            yield RawSample(
                    content=raw_sample_dict["content"],  # type: ignore
                    id=int(raw_sample_dict["id"]), # type: ignore
                    label=raw_sample_dict["label"] if "label" in raw_sample_dict.keys() else None, # type: ignore
                    probs=raw_sample_dict["probs"] if "probs" in raw_sample_dict.keys() else [] # type: ignore
                )
            
            





    
    
    