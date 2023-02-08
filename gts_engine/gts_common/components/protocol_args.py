"""工程协议参数集合模块.

Todo:
    - [ ] (Jiang Yuzhen) 本模块是为了将gts-factory的通用调用协议参数模块化，整合进gts-engine后，
        需要重新考虑存在的必要性。
"""
from pathlib import Path
from typing import Optional

from gts_common.framework import BaseArguments
from gts_common.framework.consts import TRAIN_MODE
from gts_common.utils.path import mk_inexist_dir
from pydantic import DirectoryPath


class ProtocolArgs(BaseArguments):
    """通用的工程协议参数集合."""

    input_dir: DirectoryPath
    """输入数据路径"""
    pretrained_model_root: DirectoryPath
    """预训练模型路径"""
    output_dir: DirectoryPath
    """输出路径"""
    train_level: TRAIN_MODE
    """训练模式"""

    def _add_args(self, parser) -> None:
        parser.add_argument("--gts_input_path",
                            type=Path,
                            dest="input_dir",
                            help="训练数据集路径",
                            required=True)
        parser.add_argument("--gts_pretrained_model_path",
                            type=Path,
                            dest="pretrained_model_root",
                            help="预训练模型根目录",
                            required=True)
        parser.add_argument("--gts_output_dir",
                            type=Path,
                            dest="output_dir",
                            help="输出文件路径",
                            required=True)
        parser.add_argument(
            "--gts_train_level",
            type=TRAIN_MODE,
            choices=TRAIN_MODE,
            dest="train_level",
            help="运行模式: [0 - default | 1 - student | 2 - gts | 3 - 快速模式]",
            required=True)

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
    """使当前参数包含ProtocolArgs并隔离.

    使参数foo可以直接通过args.foo而非args.protocol_args.foo访问
    """

    protocol_args: ProtocolArgs
    """通用工程协议"""

    # ========================== 隔离通用工程协议接口 ===============================

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
