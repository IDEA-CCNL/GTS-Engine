"""EDA数据增强模块."""
import copy
import multiprocessing as mp
import os
import time
from typing import Generic, List, Optional, Protocol, Sequence, TypeVar, Union

from gts_common.framework.mixin import OptionalLoggerMixin
from gts_common.utils.json_utils import dump_json_list, load_json_list
from pydantic import FilePath
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer


class SampleProto(Protocol):
    """数据增强样本类型协议.

    至少需要包含可读写的text属性
    """

    @property
    def text(self) -> str:
        ...

    @text.setter
    def text(self, value: str) -> None:
        ...


_SampleType = TypeVar("_SampleType", bound=SampleProto)


class EDA(OptionalLoggerMixin, Generic[_SampleType]):
    """EDA数据增强模块.

    Example:
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class Sample:
        ...     text: str
        ...
        >>> eda = EDA[Sample](logger_name=None)
        >>> sample_list = [
        ...     Sample(text="示例数据1"),
        ...     Sample(text="示例数据2")
        ... ]
        >>> eda_sample_list = eda.eda_aug(sample_list=sample_list, aug_path="./eda_aug.json")
        >>> eda_sample_list
        [Sample(text='配置文件数据1'), Sample(text='实例数据1'), Sample(text='数据1'), Sample(text='1'), Sample(text='示例数据1'), Sample(text='示例信息2'), Sample(text='示例统计数据2'), Sample(text='示例数据2')]
    """

    def __init__(self,
                 alpha: float = 0.1,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 logger_name: Optional[str] = None):
        """实例化EDA.

        Args:
            tokenizer (PreTrainedTokenizer):
                TODO (Jiang Yuzhen) 参数没有用到，后续移除
            alpha (float, optional):
                TODO (Jiang Yuzhen) 参数没有用到，后续移除
            logger_name (Optional[str], optional):
                用于输出信息的logger全局名称，传入None则使用print输出。默认为None。
        """
        # textda加载时间较长，放在类初始化时进行
        from textda.data_expansion import data_expansion
        self.__data_expansion = data_expansion
        OptionalLoggerMixin.__init__(self, logger_name)

    def eda_aug(self,
                sample_list: Sequence[_SampleType],
                aug_path: Union[FilePath, str],
                aug_num: int = 10) -> List[_SampleType]:
        """使用eda进行数据增强.

        Args:
            sample_list (Sequence[_SampleType]): 需要增强的样本列表
            aug_path (Union[FilePath, str]): 数据增强缓存文件路径
            aug_num (int, optional):
                TODO (Jiang Yuzhen) 参数没有用到，后续移除

        Returns:
            Sequence[_SampleType]: eda增强后的样本列表
        """
        self.__sample_cls = type(sample_list[0])
        if os.path.exists(aug_path):
            # 缓存文件存在，则直接读取
            self.info("EDA file exists, load data...")
            return self.__load_data(aug_path)
        else:
            # 缓存文件不存在，则进行增强
            self.info("generating EDA data...")
            start = time.time()
            eda_sample_list = self.__generate_data(sample_list, aug_num)
            dump_json_list(eda_sample_list, aug_path)
            end = time.time()
            self.info(
                f"generating EDA data...finished, consuming {end - start:.4f}s"
            )
            return eda_sample_list

    # ========================== private ===============================

    def __load_data(self, aug_path: Union[FilePath, str]) -> List[_SampleType]:
        eda_sample_list = list(
            load_json_list(aug_path, type_=self.__sample_cls))
        return eda_sample_list

    def __generate_data(self,
                        sample_list: Sequence[_SampleType],
                        aug_num: int = 10) -> List[_SampleType]:
        res: List[_SampleType] = []
        for sample in sample_list:
            res += self._process_sample(sample=sample)
        # pool = mp.Pool(processes=8)
        # iters = pool.imap(self._process_sample, sample_list)
        # with tqdm(total=len(sample_list), desc="EDA Augmentation") as p_bar:
        #     for eda_sample_list in iters:
        #         res += eda_sample_list
        #         p_bar.update(1)
        #     self.info(p_bar.__str__())
        # pool.close()
        # pool.join()
        return res

    def _process_sample(self,
                        sample: _SampleType,
                        aug_num: int = 10) -> List[_SampleType]:

        def create_eda_sample(new_text: str) -> _SampleType:
            new_sample = copy.deepcopy(sample)
            new_sample.text = new_text
            return new_sample

        return [
            create_eda_sample(text)
            for text in self.__data_expansion(sample.text, num_aug=aug_num)
        ]
