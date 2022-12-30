from typing import Optional, List, Union
import os
import time
from pathlib import Path
from pydantic import DirectoryPath, FilePath

from ..base_arguments import BaseArguments, GeneralParser
from ...components import ProtocolArgsMixin
from ...utils.path import mk_inexist_dir
from ...components.lightning_callbacks.adaptive_val_intervals import ADAPTIVE_VAL_INTERVAL_MODE
from ..consts import RUN_MODE

class BaseTrainingArgumentsClf(BaseArguments, ProtocolArgsMixin): 

    #############################################################################################
    ######################################## 外部参数 ##########################################
    
    dataset: str
    gpu_num: int
    load_data_ratio: float
    selected_pretrained_model_dir: Optional[DirectoryPath]
    """指定的预训练模型路径"""
    timestamp: str
    debug: bool
    """是否为debug模式"""
    run_mode: RUN_MODE
    num_workers: int
    train_batchsize_per_device: int
    valid_batchsize_per_device: int
    test_batchsize_per_device: int
    max_length: int
    seed: int
    learning_rate: float
    train_data_path: FilePath
    """训练数据集"""
    dev_data_path: Optional[FilePath]
    """验证数据集"""
    test_data_path: Optional[FilePath]
    """离线测试数据集"""
    online_test_data_path: Optional[FilePath]
    """无标注推理数据集"""
    label2id_path: FilePath
    """label2id文件"""
    _log_dir: Optional[DirectoryPath]

    @property
    def log_dir(self) -> DirectoryPath:
        return self.log_dir if self._log_dir is None else self._log_dir

    def _add_args(self, parser: GeneralParser) -> None:
        parser.add_argument("--train_data_path", dest="train_data_path", type=Path, help="指定训练数据集路径", required=True)
        parser.add_argument("--dev_data_path", dest="dev_data_path", type=Path, default=None, help="[可选]指定验证数据集路径")
        parser.add_argument("--test_data_path", dest="test_data_path", type=Path, default=None, help="[可选]指定离线测试数据集路径")
        parser.add_argument("--online_test_data_path", dest="online_test_data_path", type=Path, default=None, help="[可选]指定在线测试数据集路径")
        parser.add_argument("--dataset", dest="dataset", type=str, help="数据集名称/路径名", default="default")
        parser.add_argument("--label2id_path", dest="label2id_path", type=Path, help="指定label2id文件路径", required=True)
        parser.add_argument("--log_dir", dest="_log_dir", type=Path, default=None, help="[可选]指定日志文件保存路径")
        parser.add_argument("--run_mode", dest="run_mode", type=RUN_MODE, 
                            choices=RUN_MODE, default=RUN_MODE.ONLINE, 
                            help="运行模式 [offline - 离线测试 | online - 上线运行]")
        parser.add_argument("--load_data_ratio", dest="load_data_ratio", type=float, default=1, help="[可选]指定加载数据比例（0到1之间），用于测试")
        parser.add_argument("--selected_pretrained_model_dir", dest="selected_pretrained_model_dir", type=str, default=None, help="[可选]指定预训练模型路径")
        parser.add_argument("--seed", dest="seed", type=int, default=42)
        parser.add_argument("--debug", action="store_true", help="[可选]debug模式")
        parser.add_argument("--num_workers", dest="num_workers", type=int, default=6)
        parser.add_argument("--train_batchsize", dest="train_batchsize_per_device", type=int, default=8)
        parser.add_argument("--valid_batchsize", dest="valid_batchsize_per_device", type=int, default=8)
        parser.add_argument("--test_batchsize", dest="test_batchsize_per_device", type=int, default=8)
        parser.add_argument("--max_length", dest="max_length", type=int, default=512)
        parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=2e-5)
        parser.add_argument("--gpu_num", dest="gpu_num", type=int, default=1, help="[可选]指定训练gpu数量，系统自动分配空闲gpu，-1为使用全部可见gpu，默认为1")
        
    #############################################################################################
    ######################################## 固定参数 ##########################################
    
    pretrained_model_dir: DirectoryPath
    logger = "fintune_logger"
    prefix_prompt = "文本分类任务："
    training_label_prompt = "[unused1][unused2][MASK][MASK][MASK][unused3][unused4]"
    inference_label_prompt = "[unused1][MASK][MASK][MASK][MASK][MASK][unused2]"
    label_guided_rate = 0.5
    wwm_mask_rate = 0.12
    @property
    def train_batch_size(self) -> int:
        return self.train_batchsize_per_device * self.device_num
    @property
    def test_batch_size(self) -> int:
        return self.test_batchsize_per_device * self.device_num
    @property
    def valid_batch_size(self) -> int:
        return self.valid_batchsize_per_device * self.device_num
    epoch = 5
    @property
    def decay_epoch(self): 
        return self.epoch - 1
    warm_up_epoch = 1
    clip_norm = 0.25
    validation_mode = ADAPTIVE_VAL_INTERVAL_MODE.ADAPTIVE
    timestamp = str(int(time.time()))
    
    @property
    def ft_output_dir(self) -> DirectoryPath:
        if self.run_mode == RUN_MODE.OFFLINE:
            output_path = self.protocol_args.output_dir / self.dataset
            output_path = output_path / self.timestamp
        else:
            output_path = self.protocol_args.student_output_dir / 'finetune_output'
        return output_path
    
    @property
    def device_num(self) -> int:
        return self.gpu_num
    
    #############################################################################################
    ######################################## 其他逻辑 ##########################################
    
    def _after_parse(self) -> None:
        mk_inexist_dir(self.log_dir)
        mk_inexist_dir(self.ft_output_dir, clean=True)
     
   
    
class BaseInferenceArgumentsClf(BaseArguments):
    
    model_save_dir: DirectoryPath
    label2id_path: FilePath
    
    def _add_args(self, parser) -> None:
        parser.add_argument("--model_save_dir", dest="model_save_dir", type=Path, help="输出文件路径", required=True)
        parser.add_argument("--label2id_path", dest="label2id_path", type=Path, help="[可选]指定label2id文件路径", required=True)
        
    logger = "fintune_logger"
    prefix_prompt = "文本分类任务："
    inference_label_prompt = "[unused1][MASK][MASK][MASK][MASK][MASK][unused2]"
    max_length = 512
    batch_size = 8
    
    @property
    def model_state_dict_file_path(self) -> FilePath:
        return self.model_save_dir / "finetune_pytorch.bin"
    
    @property
    def pretrained_model_dir(self) -> DirectoryPath:
        return self.model_save_dir