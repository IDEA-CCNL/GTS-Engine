from typing import Literal, Optional
import os
from pathlib import Path
from pydantic import FilePath, DirectoryPath

from ..framework import BaseArguments
from ..framework.consts import TRAIN_MODE
from ..utils.path import mk_inexist_dir

class ProtocolArgs(BaseArguments):
    input_dir: DirectoryPath
    pretrained_model_root: DirectoryPath
    output_dir: DirectoryPath
    train_level: TRAIN_MODE
    
    def _add_args(self, parser) -> None:
        parser.add_argument("--gts_input_path", type=Path, dest="input_dir", help="训练数据集路径", required=True)
        parser.add_argument("--gts_pretrained_model_path", type=Path, dest="pretrained_model_root", help="预训练模型根目录", required=True)
        parser.add_argument("--gts_output_dir", type=Path, dest="output_dir", help="输出文件路径", required=True)
        parser.add_argument("--gts_train_level", type=TRAIN_MODE, choices=TRAIN_MODE, dest="train_level", 
                            help="运行模式: [0 - default | 1 - student | 2 - gts | 3 - 快速模式]", required=True)
        
    def _after_parse(self) -> None:
        mk_inexist_dir(self.output_dir)
        
    @property
    def _arg_name(self) -> Optional[str]:
        return "调用协议参数"
    
    @property
    def student_output_dir(self) -> DirectoryPath:
        return self.output_dir / "student_output"
    
    @property
    def log_dir(self) -> DirectoryPath:
        return self.output_dir / "logs"

class ProtocolArgsMixin:
    """使当前参数包含ProtocolArgs并隔离"""
    
    protocol_args: ProtocolArgs
    """通用工程协议"""
    
    ################## 隔离通用工程协议接口 ################
    
    @property
    def input_dir(self):
        return self.protocol_args.input_dir
    
    @property
    def pretrained_model_root(self):
        return self.protocol_args.pretrained_model_root
    
    @property
    def student_output_dir(self):
        return self.protocol_args.student_output_dir
    
    @property
    def log_dir(self):
        return self.protocol_args.log_dir
    
    @property
    def train_level(self):
        return self.protocol_args.train_level
    
    @property
    def output_dir(self):
        return self.protocol_args.output_dir
    
    