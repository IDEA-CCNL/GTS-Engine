from typing import Generator, Optional, List, Dict, Protocol, Sequence, Type, TypeVar, Generic, Union
import os
from transformers.tokenization_utils import PreTrainedTokenizer
from dataclasses import asdict, dataclass
import multiprocessing as mp
from tqdm import tqdm
import time
import copy
from pydantic import FilePath

from ..utils.json_processor import load_json_list, dump_json_list
from .text_tools import segment_text
from ..framework.mixin import OptionalLoggerMixin

class SampleProto(Protocol):
    """传入的数据需要有text字段"""
    text: str
    
_SampleType = TypeVar("_SampleType", bound=SampleProto)

class EDA(OptionalLoggerMixin, Generic[_SampleType]):
    #############################################################################################
    ######################################## public ##########################################
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        alpha: float = 0.1, 
        logger_name: Optional[str] = None
    ):
        from textda.data_expansion import data_expansion # 加载时间较长，放在类初始化时进行
        self.__data_expansion = data_expansion
        OptionalLoggerMixin.__init__(self, logger_name)
        self._alpha = alpha
        self._tokenizer = tokenizer
    
    def eda_aug(self, sample_list: Sequence[_SampleType], aug_path: Union[FilePath, str], aug_num: int = 10) -> Sequence[_SampleType]:
        self.__sample_cls = type(sample_list[0])
        if os.path.exists(aug_path):
            self.info("EDA file exists, load data...")
            return self.__load_data(aug_path)
        else:
            self.info("generating EDA data...")
            start = time.time()
            eda_sample_list = self.__generate_data(sample_list, aug_num)
            dump_json_list(eda_sample_list, aug_path)
            end = time.time()
            self.info(f"generating EDA data...finished, consuming {end - start:.4f}s")
            return eda_sample_list
        
    #############################################################################################
    ######################################## private ##########################################    
    
    def __load_data(self, aug_path: Union[FilePath, str]) -> List[_SampleType]:
        eda_sample_list = list(load_json_list(aug_path, type_=self.__sample_cls))
        return eda_sample_list
    
    def __generate_data(self, sample_list: Sequence[_SampleType], aug_num: int = 10) -> List[_SampleType]:
        res: List[_SampleType] = []
        pool = mp.Pool(processes=8)
        iters = pool.imap(self._process_sample, sample_list)
        with tqdm(total=len(sample_list), desc="EDA Augmentation") as p_bar:
            for eda_sample_list in iters:
                res += eda_sample_list
                p_bar.update(1)
            self.info(p_bar.__str__())
        pool.close()
        pool.join()
        return res
        
    def _process_sample(self, sample: _SampleType, aug_num: int = 10) -> List[_SampleType]:
        def create_eda_sample(new_text: str) -> _SampleType:
            new_sample = copy.deepcopy(sample)
            new_sample.text = new_text
            return new_sample
            
        return [create_eda_sample(text) for text in self.__data_expansion(sample.text, num_aug=aug_num)]