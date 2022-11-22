from typing import List, Literal
from argparse import Namespace
from enum import Enum

Path = str

class GTSEngineArgs(Namespace):
    engine_type: Literal["qiankunding", "bagualu"]
    train_mode: Literal["fast", "standard", "advanced"]
    task_dir: Path
    task_type: Literal["classification", "similarity", "nli"]
    num_workers: int
    train_batchsize: int
    valid_batchsize: int
    test_batchsize: int
    max_len: int
    pretrained_model_dir: Path
    data_dir: Path
    train_data: Path
    valid_data: Path
    test_data: Path
    label_data: Path
    save_path: Path
    seed: int
    lr: float
    gpus: int
    num_sanity_val_steps: int
    accumulate_grad_batches: int
    val_check_interval: float
    
class TRAIN_MODE(Enum):
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
