"""模型训练pipeline基类模块.

Todo:
    目前存在最主要的问题，在于训练阶段的划分粒度太大，不利于任务层抽象。典型的例子如clf_std，
    在_after_training()阶段内部需要插入很简单的一小段knn的逻辑，却不得不将_after_training()
    整个重写一遍，但是绝大部分代码都是一样的，就违背了框架的初衷。目前能想到的改进方法如下:

    - [ ] (Jiang Yuzhen) 拆分细化训练阶段的接口，如将_after_training()继续拆分为
        模型选择阶段、模型评估阶段、模型保存阶段、垃圾处理阶段等，减小开发者重写函数的体量。
    - [ ] (Jiang Yuzhen)  在拆分流程后，进一步考虑改用回调的设计模式，将各个处理流程单
        元通过回调函数来模块化，使开发者可以通过定义回调函数来方便客制化功能，而不是大量重
        复的函数重写。但也相应地提升了框架设计和开发难度，特别是回调函数之间信息传递的设计，
        需要权衡需求与成本。

    另外，使用回调函数的方式会让人疑惑gts-engine和类似pytorch_lightning这样的训练框架之间
    的界限。个人的看法是，gts-engine做的是对具体的完整任务的封装，负责的是任务从启动和输入、
    文件读取一直到交给用户手中的结果的全流程，而pytorch_lightning仅仅是针对模型训练，所以两
    者并不在一个层级上，就像pytorch_lightning和pytorch不在一个层级一样。
"""
from abc import ABCMeta, abstractmethod
from typing import List, Optional

from .mixin import ArgsMixin


class BaseTrainingPipeline(ArgsMixin, metaclass=ABCMeta):
    """模型训练Pipeline接口基类.

    定义了模型训练Pipeline对外暴露的启动接口main()，和子类应当实现的接口函数，来定义模型训练的
    不同阶段。同时通过继承`ArgsMixin`来提供和参数集合的交互，需要对该类对应的参数类进行声明，详
    见`ArgsMixin`文档。
    """

    def __init__(self, args_parse_list: Optional[List[str]] = None):
        """实例化训练Pipeline.

        Args:
            args_parse_list (Optional[List[str]], optional):
                训练参数列表，设置为None则从命令行获取。 Defaults to None.
        """
        ArgsMixin.__init__(self, args_parse_list)

    @abstractmethod
    def _before_training(self) -> None:
        """训练前处理."""
        pass

    @abstractmethod
    def _train(self) -> None:
        """模型训练."""
        pass

    @abstractmethod
    def _after_training(self) -> None:
        """训练后处理."""
        pass

    def main(self):
        """启动训练pipeline."""
        self._before_training()
        self._train()
        self._after_training()
