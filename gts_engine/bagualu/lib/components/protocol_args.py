from typing import Literal, Optional
import os

from ..framework import BaseArguments
from ..framework.consts import TRAIN_MOD
from ..utils.path import mk_inexist_dir

class ProtocolArgs(BaseArguments):
    input_dir: str
    pretrained_model_root: str
    output_dir: str 
    
    @property
    def train_level(self) -> TRAIN_MOD:
        """运行模式:  
        0 - default | 1 - student | 2 - gts | 3 - 快速模式
        """
        return TRAIN_MOD(self._train_level)
    _train_level: int
    
    def _add_args(self, parser) -> None:
        parser.add_argument("--gts_input_path", type=str, dest="input_dir", help="训练数据集路径", required=True)
        parser.add_argument("--gts_pretrained_model_path", type=str, dest="pretrained_model_root", help="预训练模型根目录", required=True)
        parser.add_argument("--gts_output_dir", type=str, dest="output_dir", help="输出文件路径", required=True)
        parser.add_argument("--gts_train_level", type=int, choices=[item.value for item in TRAIN_MOD], dest="_train_level", 
                            help="运行模式: [0 - default | 1 - student | 2 - gts | 3 - 快速模式]", required=True)
        
    def _after_parse(self) -> None:
        mk_inexist_dir(self.output_dir)
        
    @property
    def _arg_name(self) -> Optional[str]:
        return "调用协议参数"
    
    @property
    def student_output_dir(self) -> str:
        return os.path.join(self.output_dir, "student_output")
    
    @property
    def log_dir(self) -> str:
        return os.path.join(self.student_output_dir, "logs")

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
    
    