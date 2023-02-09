"""常量定义模块.

将所有需要复用的常量定义在这里，防止循环导入。
"""
from enum import Enum
from typing import Optional

from torch import Tensor
from typing_extensions import TypedDict

# =============================================================================
# types
# =============================================================================


class BertInput(TypedDict):
    """Bert模型输入字段."""
    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor
    labels: Optional[Tensor]


# =============================================================================
# enums
# =============================================================================


class TRAIN_MODE(Enum):
    DEFAULT = "0"
    STUDENT = "1"
    GTS = "2"
    FAST = "3"


class RUN_MODE(Enum):
    OFFLINE = "offline"
    ONLINE = "online"


class TASK(Enum):
    SINGLE_SENTENCE_CLF = 1
    SENTENCE_CLF = 2
    MULTI_TASK = 3


class TRAINING_STAGE(Enum):
    TRAINING = 1
    TEST = 2
    VALIDATION = 3
    INFERENCE = 4
