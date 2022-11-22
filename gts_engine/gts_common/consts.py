from typing import List, Literal, Optional
from argparse import Namespace
from enum import Enum
import os
from pathlib import Path

PathStr = str

class GTSEngineArgs(Namespace):
    engine_type: Literal["qiankunding", "bagualu"]
    train_mode: Literal["fast", "standard", "advanced"]
    task_dir: PathStr
    task_type: Literal["classification", "similarity", "nli"]
    num_workers: int
    train_batchsize: int
    valid_batchsize: int
    test_batchsize: int
    max_len: int
    pretrained_model_dir: PathStr
    data_dir: PathStr
    train_data: str
    valid_data: str
    test_data: str
    label_data: str
    save_path: PathStr
    seed: int
    lr: float
    gpus: int
    num_sanity_val_steps: int
    accumulate_grad_batches: int
    val_check_interval: float
    
    @property
    def train_data_path(self) -> Path:
        return Path(self.data_dir) / self.train_data
    
    @property
    def valid_data_path(self) -> Path:
        return Path(self.data_dir) / self.valid_data
    
    @property
    def test_data_path(self) -> Optional[Path]:
        return Path(self.data_dir) / self.test_data if hasattr(self, "test_data") else None
    
    @property
    def label_data_path(self) -> Optional[Path]:
        return Path(self.data_dir) / self.label_data if hasattr(self, "label_data") else None
    
class TRAIN_MODE(Enum):
    DEFAULT = "default"
    STD = "standard"
    FAST = "fast"
    ADV = "advanced"

class TASK(Enum):
    CLS = "classification"
    NLI = "nli"
    SIM = "similarity"
    IE = "ie"
    
class ENGINE_TYPE(Enum):
    QKD = "qiankunding"
    BGL = "bagualu"
