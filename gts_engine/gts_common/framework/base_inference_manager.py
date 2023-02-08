"""推理器接口基类模块.

加载训练好的模型，进行推理任务

Todo:
    - [ ] (Jiang Yuzhen) 始终觉得manager的命名有点怪，可以再考虑一下命名
"""
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel

from .mixin import ArgsMixin

# 因为是向外暴露的接口，希望可以对推理样本的类型进行runtime检查，所以使用pydantic的结构体
InfSample = BaseModel
# 推理输出是一个字典
InfOutput = Dict


class BaseInferenceManager(ArgsMixin, metaclass=ABCMeta):
    """推理器接口基类.

    描述了推理器应当对外暴露的接口待子类实现。需要注意的是，推理器继承了`ArgsMixin`来提供
    和参数集合的交互，需要对该类对应的参数类进行声明，详见`ArgsMixin`文档。
    """

    def __init__(self, args_parse_list: Optional[List[str]] = None):
        """实例化推理器.

        Args:
            args_parse_list (Optional[List[str]], optional):
                推理器参数列表，为None时则从命令行获取. Defaults to None.
        """
        ArgsMixin.__init__(self, args_parse_list)

    @abstractmethod
    def prepare_inference(self) -> None:
        """推理准备.

        所有与具体样本推理无关的、一次性的推理准备逻辑都放在这里实现，如加载模型等。
        """

    @abstractmethod
    def inference(self, sample: Sequence[InfSample]) -> InfOutput:
        """推理

        对具体样本进行推理。在完成推理器的实例化和准备后，可以反复调用。

        Args:
            sample (Sequence[InfSample]):
                需要推理的样本序列，样本字段需要使用pydantic.BaseModel进行定义以支持动态类型检查。

        Returns:
            InfOutput: 推理结果字典，待子类根据任务需要进一步定义。
        """
