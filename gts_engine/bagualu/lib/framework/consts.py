from typing import NewType, NamedTuple, Dict, List, Optional, Protocol, Union
from typing_extensions import TypedDict
from enum import Enum
from torch import Tensor

#############################################################################################
## Types
#############################################################################################

class BertInput(TypedDict):
    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor
    labels: Optional[Tensor]

#############################################################################################
## Enums
#############################################################################################

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