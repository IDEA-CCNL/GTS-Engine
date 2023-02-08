"""框架包.

理想中的框架分为三层：
    * 通用接口层，如base_training_pipeline、base_inference_manager等。定义各个基本模块的性质
      和通用行为，向下层提供基本的功能如与参数集合的交互等。
    * 任务层，如classification_finetune。基于base层，定义特定任务的通用行为，因为特定任务之
      所以为其所是，一定是因为其有特定的输入输出内容和格式、相似的基本处理流程，只是使用了不同的
      模型、有差异的模型训练方式和数据处理逻辑，所以将任务相关的共性定义在此层，而将具体的模型训练
      和数据处理的实现交给下一层，以方便针对已有任务扩展新的实现方式。
    * 实现/模式层，某一任务的特定一种实现（模式），如classification的std、adv等模式，通过实现
      任务层提供的数据处理、模型结构定义、模型训练方式定义等接口，来定义一个具体的实现。

但是在具体实践中，不一定要有中间的任务层，任务层存在的前提是这个任务存在多种实现，且实现中存在共性。所以
在任务尚未需要多种实现时，可以从通用接口层直接继承实现，在需要有多种实现时再对任务进行进一步抽象。

包含:
    * BaseArguments: 参数集合基类
    * BaseTrainingPipeline: 通用训练Pipeline接口基类
    * BaseInferenceManager: 通用推理器接口基类
    * BaseGtsEngineInterface: 用于嵌入gts-engine的胶水层基类
    * GtsEngineArgs: gts-engine参数集合
    * mixin: 提供小单位功能的Mixin集合
    * classification_finetune: 句子分类任务层基类集合

Todo:
    - [ ] (Jiang Yuzhen) 将classification_finetune改为classification，作为所有文本分类的基类集合
    - [ ] (Jiang Yuzhen) base_arguments与业务相对无关，考虑放进utils中
"""
from .base_arguments import BaseArguments
from .base_gts_engine_interface import BaseGtsEngineInterface, GtsEngineArgs
from .base_inference_manager import BaseInferenceManager
from .base_training_pipeline import BaseTrainingPipeline

__all__ = [
    "BaseArguments", "BaseTrainingPipeline", "BaseInferenceManager",
    "GtsEngineArgs", "BaseGtsEngineInterface"
]
