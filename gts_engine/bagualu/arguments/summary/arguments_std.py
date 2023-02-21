# Copyright 2021 The IDEA Authors. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional

from gts_common.components.protocol_args import ProtocolArgsMixin
from gts_common.framework.base_arguments import BaseArguments, GeneralParser
from gts_common.utils.path import mk_inexist_dir


class TrainingArgumentsSummaryStd(BaseArguments, ProtocolArgsMixin):
    """摘要生成finetune参数."""

    # 输出相关参数
    @property
    def ft_output_dir(self) -> str:
        """finetune输出路径."""
        return os.path.join(self.protocol_args.output_dir)

    @property
    def log_dir(self) -> str:
        return os.path.join(self.ft_output_dir, "log")

    @property
    def ckpt_dirpath(self) -> str:
        """checkpoint dirpath."""
        return os.path.join(self.ft_output_dir, "checkpoints")

    @property
    def tensorboard_dir(self) -> str:
        """tensorboard日志输出路径."""
        return os.path.join(self.ft_output_dir, "tensorboard")

    @property
    def prediction_save_dir(self) -> str:
        """预测结果保存路径."""
        return os.path.join(self.ft_output_dir, "predictions")

    @property
    def model_save_dir(self) -> str:
        """模型保存路径."""
        return os.path.join(self.ft_output_dir, "model")

    @property
    def best_ckpt_path(self) -> str:
        """最优checkpoint路径."""
        return os.path.join(self.model_save_dir, "best.ckpt")

    @property
    def last_ckpt_path(self) -> str:
        """最后一个checkpoint路径."""
        return os.path.join(self.ckpt_dirpath, "last.ckpt")

    # 数据集相关参数

    _train_data_path: Optional[str]
    _dev_data_path: Optional[str]
    _test_data_path: Optional[str]
    _online_test_data_path: Optional[str]

    @property
    def train_data_path(self) -> str:
        """训练集."""
        default_path = os.path.join(self.protocol_args.input_dir,
                                    "train.jsonl")
        return self._train_data_path if self._train_data_path else default_path

    @property
    def dev_data_path(self) -> str:
        """验证集."""
        default_path = os.path.join(self.protocol_args.input_dir, "val.jsonl")
        return self._dev_data_path if self._dev_data_path else default_path

    @property
    def test_data_path(self) -> str:
        """测试集."""
        default_path = os.path.join(self.protocol_args.input_dir, "test.jsonl")
        return self._test_data_path if self._test_data_path else default_path

    @property
    def online_test_data_path(self) -> str:
        """无标注数据."""
        default_path = os.path.join(self.protocol_args.input_dir,
                                    "test_online.jsonl")
        return self._online_test_data_path if self._online_test_data_path else default_path

    # 固定参数
    logger: str = "summary_logger"
    batch_size: int = 16
    num_workers: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup: float = 0.06
    loss: str = "dc"
    loss_boost: float = 100000.0
    log_step: int = 20
    dc_smooth: float = 1.0
    neg_drop: float = 0.4
    distill_self: bool = False
    sep_data: bool = False
    sep_need_ori: bool = False
    sep_need_neg: bool = False

    cut_text: bool = True
    cut_text_len: int = 400
    cut_text_stride: int = 200

    # trainer参数
    max_steps: int = -1
    max_epochs: int = 50
    min_epochs: int = 1
    precision: int = 16
    check_val_every_n_epoch: int = 1
    gradient_clip_val: float = 1.0
    val_check_interval: float = 0.5
    enable_progress_bar: bool = False
    accumulate_grad_batches: int = 1
    accelerator: str = "gpu"

    # checkpoints参数
    ckpt_monitor: str = "dev_rougeL_fmeasure"
    ckpt_mode: str = "max"
    ckpt_save_last: bool = True
    ckpt_filename: str = "model-{epoch:02d}-{dev_rougeL_fmeasure:.4f}"
    ckpt_save_top_k: int = 3
    ckpt_every_n_train_steps: int = 100
    ckpt_save_weights_only: bool = True

    # 传入参数
    dataset: str
    gpus: int
    max_enc_length: int
    max_dec_length: int
    rouge_keys: str

    def _add_args(self, parser: GeneralParser) -> None:
        # 数据集参数
        parser.add_argument("--dataset",
                            dest="dataset",
                            type=str,
                            help="数据集名称",
                            default="default")
        parser.add_argument("--train_data_path",
                            dest="_train_data_path",
                            type=str,
                            default=None,
                            help="[可选]指定训练集路径")
        parser.add_argument("--dev_data_path",
                            dest="_dev_data_path",
                            type=str,
                            default=None,
                            help="[可选]指定验证集路径")
        parser.add_argument("--test_data_path",
                            dest="_test_data_path",
                            type=str,
                            default=None,
                            help="[可选]指定测试集路径")
        parser.add_argument("--online_test_data_path",
                            dest="_online_test_data_path",
                            type=str,
                            default=None,
                            help="[可选]指定在线测试数据路径")

        # 训练参数
        parser.add_argument("--gpus",
                            dest="gpus",
                            type=str,
                            default=None,
                            help="[可选]指定gpu个数")
        parser.add_argument("--learning_rate",
                            dest="learning_rate",
                            type=float,
                            default=1e-5,
                            help="[可选]学习率，默认1e-5")
        parser.add_argument("--accumulate_grad_batches",
                            dest="accumulate_grad_batches",
                            type=int,
                            default=1,
                            help="[可选]梯度累积，默认1")
        parser.add_argument("--num_workers",
                            dest="num_workers",
                            type=int,
                            default=8,
                            help="[可选]工作数，默认8")
        parser.add_argument("--max_enc_length",
                            dest="max_enc_length",
                            type=int,
                            default=512,
                            help="[可选]最大长度，默认512")
        parser.add_argument("--max_dec_length",
                            dest="max_dec_length",
                            type=int,
                            default=256,
                            help="[可选]最大长度，默认256")
        parser.add_argument("--val_check_interval",
                            dest="val_check_interval",
                            type=float,
                            default=0.5,
                            help="[可选]验证间隔，默认0.5")
        parser.add_argument("--batch_size",
                            dest="batch_size",
                            type=int,
                            default=16,
                            help="[可选]train batch大小，默认16")
        parser.add_argument("--max_epochs",
                            dest="max_epochs",
                            type=int,
                            default=None,
                            help="[可选]训练最大epoch数")
        parser.add_argument("--min_epochs",
                            dest="min_epochs",
                            type=int,
                            default=None,
                            help="[可选]训练最小epoch数")
        parser.add_argument("--rouge_keys",
                            dest="rouge_keys",
                            type=str,
                            default='rougeL,rouge1,rouge2',
                            help="[可选]计算rouge的指标，默认rougeL,rouge1,rouge2")

    def _after_parse(self) -> None:
        mk_inexist_dir(self.ft_output_dir)
        mk_inexist_dir(self.log_dir)
        mk_inexist_dir(self.ckpt_dirpath, clean=True)
        mk_inexist_dir(self.tensorboard_dir)
        mk_inexist_dir(self.prediction_save_dir, clean=True)
        mk_inexist_dir(self.model_save_dir, clean=True)


class InferenceArgumentsSummaryStd(BaseArguments):
    """摘要生成finetune参数."""

    task_dir: str
    pretrained_model_root: str

    num_workers: int = 8
    batch_size: int = 8
    max_enc_length: int = 128
    max_dec_length: int = 64
    rouge_keys: str = 'rougeL,rouge1,rouge2'

    @property
    def model_save_dir(self) -> str:
        """模型保存路径."""
        return os.path.join(self.task_dir, "outputs/model")

    @property
    def best_ckpt_path(self) -> str:
        """最优checkpoint路径."""
        return os.path.join(self.model_save_dir, "best.ckpt")

    def _add_args(self, parser: GeneralParser) -> None:
        # 数据集参数
        parser.add_argument("--task_dir",
                            dest="task_dir",
                            required=True,
                            type=str,
                            help="specific task directory")
        parser.add_argument("--pretrained_model_root",
                            dest="pretrained_model_root",
                            required=True,
                            type=str,
                            help="path to pretrained model")
