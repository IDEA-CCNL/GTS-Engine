from dataclasses import dataclass, field
from torch import Tensor
from typing import List, Optional, Protocol, TypedDict, Dict, NamedTuple
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer
from pydantic import BaseModel, Field

#############################################################################################
## Sample & Data
#############################################################################################

class RawSample(BaseModel):
    """数据在json文件中储存的字段"""
    content: str
    id: int
    label: Optional[str] = None
    probs: List[float] = Field(default_factory=list) # 默认为空列表，dataclass不允许默认值为动态数据结构[]，故采用这种方式
    
@dataclass
class LabeledSample:
    """数据被加载并初步处理后的字段"""
    text: str
    id: int
    label: str
    label_id: int
    label_id_clf: int
    soft_label: List[float] = field(default_factory=list)

@dataclass
class UnlabeledSample:
    """无标注数据"""
    text: str
    id: int
    
@dataclass
class PreEncodedTrainSample:
    text: str
    id: int
    label: str
    label_id: int
    label_id_clf: int
    words: List[str]
    inference_prompt: List[str]
    training_prompt: List[str]
    soft_label: List[float] = field(default_factory=list)

@dataclass
class EncodedTrainSample:
    input_ids: Tensor
    input_seg: Tensor
    input_mask: Tensor
    mask_positions: Tensor
    labels: Tensor
    label_id: Tensor
    label_id_clf: Tensor
    id: int
    """数据集自带id"""
    weight: Tensor
    soft_label: Tensor
    seq_len: int
    my_id: int
    """自定义辅助id"""

@dataclass
class EncodedInfSample:
    input_ids: Tensor
    input_seg: Tensor
    input_mask: Tensor
    my_id: int
    """自定义辅助id"""
    
class InfSampleProto(Protocol):
    """用于推理的数据格式"""
    id: int
    text: str

class InferenceManagerInputSample(BaseModel):
    text: str

InferenceManagerInputSampleList = List[InferenceManagerInputSample]

#############################################################################################
## Prompt & Label
#############################################################################################

class PromptToken(NamedTuple):
    label: str
    label_id: int
    label_id_clf: int
    key: str

class PromptLabel(NamedTuple):
    label: str
    key: str

Label2Token = Dict[str, PromptToken]

class PromptLabelParseOutput(NamedTuple):
    label2token: Label2Token
    id2label: Dict[int, PromptLabel]
    label_ids: List[PromptToken]

class Label2IdValue(BaseModel):
    id: int
    label_desc_en: Optional[str] = None
    label_desc_zh: Optional[str] = None
    
Label2Id = Dict[str, Label2IdValue]

#############################################################################################
## In / Out
#############################################################################################
class TrainingModelOutput(TypedDict):
    loss_total: Tensor
    loss_ce: Tensor
    loss_mlm: Tensor
    kl_loss: Tensor
    logits: Tensor
    loss_rd: Tensor
    loss_ctr: Tensor
    loss_ner: Tensor
    
class InferenceModelOutput(TypedDict):
    positions: Tensor
    probs: Tensor
    embeds: Tensor

@dataclass
class DevOutput:
    dev_loss: float
    dev_acc: float
    
@dataclass
class PredictionResult:
    id: int
    content: str
    predict: str
    label: Optional[str] = None

@dataclass
class TrainingSettings:
    learning_rate: float
    label_guided_rate: float
    prefix_prompt: str
    training_label_prompt: str
    inference_label_prompt: str
    batch_size: int
    epoch: int
    warmup_epoch: int
    decay_epoch: int
    dropout_rate: float

class InferenceManagerOutput(TypedDict):
    predictions: List[str]
    probabilities: List[List[float]]
    
class InfBatch(TypedDict):
    input_ids: Tensor
    input_seg: Tensor
    input_mask: Tensor
    my_id: List[int]
    """自定义辅助id"""
    
class TrainBatch(TypedDict):
    input_ids: Tensor
    input_seg: Tensor
    input_mask: Tensor
    mask_positions: Tensor
    labels: Tensor
    label_id: Tensor
    label_id_clf: Tensor
    id: List[int]
    """数据集自带id"""
    weight: Tensor
    soft_label: Tensor
    seq_len: Tensor
    my_id: List[int]
    """自定义辅助id"""