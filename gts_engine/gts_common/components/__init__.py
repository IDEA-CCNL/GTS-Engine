"""开箱即用的组件包.

包含:
    * ProtocolArgs: 工程接口协议参数集合
    * ProtocolArgsMixin: 将工程接口协议参数作为子参数集合加入参数集合并隔离的Mixin
    * TokenizerGenerator: Tokenizer生成器
    * eda_utils: EDA数据增强
    * lightning_callbacks: pytorch_lightning自定义callback集
    * losses: loss计算工具集
    * metrics: 评估工具集
    * knn_for_bagualu: knn相关工具集
    * sampling_utils: 抽样、样本筛选相关工具集
    * schedulers: pytorch 学习率scheduler集
    * text_processing_tools: 文本处理工具集

Todo:
    - [ ] (Jiang Yuzhen) 一些空缺内容待补全
    - [ ] (Jiang Yuzhen) 目前觉得metrics和losses这两个包有些意义不明。一方面是losses里不只有损失计算；
        一方面metrics里一部分是pytorch模型层（logits_2_acc），一部分是模型评估工具。个人认为更好的
        处理方式是将所有可复用的pytorch模型层放在`layer`子包中，`layer`可以根据需要再划分出`losses`、
        `metrics`等，这里可以参考开源库https://github.com/shenweichen/DeepCTR/tree/master/deepctr/layers；
        再将模型评估`model_evaluation`相关的工具单独放在外面。
"""
from .eda_utils import EDA
from .protocol_args import ProtocolArgs, ProtocolArgsMixin
from .tokenizer_generator import TokenizerGenerator

__all__ = ["ProtocolArgs", "ProtocolArgsMixin", "TokenizerGenerator", "EDA"]
