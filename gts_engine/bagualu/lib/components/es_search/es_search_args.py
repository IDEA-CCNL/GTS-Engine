"""es search模块配套参数模块"""
from typing import Optional
from enum import Enum

from gts_engine.bagualu.lib.framework import BaseArguments


class SEARCH_CLIENT(Enum):
    """es搜索client"""
    WEAVIATE = 0
    """院内服务"""
    WENTIAN = 1
    """阿里问天"""


class ESSearchArgs(BaseArguments):
    """es搜索配套参数"""

    augm_es_path: Optional[str]
    """es数据增强文件缓存路径"""

    def _add_args(self, parser) -> None:
        parser.add_argument(
            "--augm_es_path", dest="augm_es_path",
            type=str, default=None, help="[可选]指定es search缓存文件路径")

    dev_step = 20
    """间隔dev_step步做一次验证"""
    required_minimum_data = 1000
    """tapt所需最少的数据量"""
    es_search_num = 20
    """es 默认检索的数据量"""
    search_batch_size = 3
    max_augment_data = 12000
    """最大检索的数据量"""
    addtion_search = 20
    """es额外需要增加的检索数据量"""
    es_search_query = 2000
    """最多检索的query的个数"""
    es_request_timeout = 1800
    """es请求超时的数据量"""
    es_num_thread = 16
    """es并行处理的线程数"""
    es_max_runtime = 90
    """es检索的"""
    stratification_sample = 1
    tapt_max_runtime = 180
    """tapt模块整体的时间"""
    add_label_description: bool = True
    start_time: float
    es_cllient = SEARCH_CLIENT.WEAVIATE

    @property
    def _arg_name(self):
        return "ES搜索参数"
