from typing import Literal, Optional
from pathlib import Path
from argparse import Namespace


PathStr = str
FileName = str

class GtsEngineArgs(Namespace):
    """GTS-Engine参数namespace"""
    engine_type: Literal["qiankunding", "bagualu"]
    train_mode: Literal["fast", "standard", "advanced"]
    task_dir: PathStr
    task_type: Literal["classification", "similarity", "nli", "ie"]
    num_workers: int
    train_batchsize: int
    valid_batchsize: int
    test_batchsize: int
    max_len: int
    pretrained_model_dir: PathStr
    data_dir: PathStr
    train_data: FileName
    valid_data: FileName
    test_data: FileName
    label_data: FileName
    save_path: PathStr
    seed: int
    lr: float
    gpus: int
    val_check_interval: float
    max_epochs: Optional[int]
    min_epochs: Optional[int]
    gradient_checkpointing_gate: str
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
