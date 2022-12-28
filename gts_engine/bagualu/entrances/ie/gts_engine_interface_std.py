# coding=utf-8
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

from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel, DirectoryPath, FilePath

from gts_common.arguments import GtsEngineArgs
from ...lib.framework.base_gts_engine_interface import (TRAIN_MODE,
                                                       BaseGtsEngineInterface)
from .training_pipeline_std import TrainingPipelineIEStd
from .inference_manager_std import InferenceManagerIEStd


class TypeCheckedTrainArgs(BaseModel):
    """ GTS-Engine相关参数进行runtime类型检查 """
    task_dir: DirectoryPath
    pretrained_model_dir: DirectoryPath
    data_dir: DirectoryPath
    save_path: DirectoryPath
    train_data_path: FilePath
    valid_data_path: FilePath
    test_data_path: Optional[FilePath]
    gpus: int
    train_mode: TRAIN_MODE
    seed: int
    lr: float
    num_workers: int
    max_len: int
    val_check_interval: float
    train_batchsize: int
    max_epochs: int
    min_epochs: int


class TypeCheckedInferenceArgs(BaseModel):
    """ GTS-Engine相关参数进行runtime类型检查 """
    task_dir: DirectoryPath
    pretrained_model_root: DirectoryPath


class GtsEngineInterfaceIEStd(BaseGtsEngineInterface):
    """ GtsEngineInterfaceIEStd """

    @property
    def _training_pipeline_type(self):
        return TrainingPipelineIEStd

    def _parse_training_args(self, args: GtsEngineArgs) -> List[str]:
        # args类型检查
        checked_args = TypeCheckedTrainArgs(
            task_dir=Path(args.task_dir),
            pretrained_model_dir=Path(args.pretrained_model_dir),
            data_dir=Path(args.data_dir),
            save_path=Path(args.save_path),
            train_data_path=args.train_data_path,
            valid_data_path=args.valid_data_path,
            test_data_path=args.test_data_path,
            gpus=args.gpus,
            train_mode=TRAIN_MODE(args.train_mode),
            seed=args.seed,
            lr=args.lr,
            num_workers=args.num_workers,
            max_len=args.max_len,
            val_check_interval=args.val_check_interval,
            train_batchsize=args.train_batchsize,
            max_epochs=args.max_epochs,
            min_epochs=args.min_epochs,
        )

        # 用户参数转化为实际参数
        args_parse_list: List[str] = []
        args_parse_list.extend(["--gts_input_path", str(checked_args.task_dir)])
        args_parse_list.extend(["--gts_pretrained_model_path",
                                str(checked_args.pretrained_model_dir)])
        args_parse_list.extend(["--gts_output_dir", str(checked_args.save_path)])
        args_parse_list.extend(["--gts_train_level", "1"])
        args_parse_list.extend(["--gpus", str(checked_args.gpus)])
        args_parse_list.extend(["--train_data_path", str(checked_args.train_data_path)])
        args_parse_list.extend(["--dev_data_path", str(checked_args.valid_data_path)])
        args_parse_list.extend(["--test_data_path", str(checked_args.test_data_path)])
        args_parse_list.extend(["--learning_rate",str(checked_args.lr)])
        args_parse_list.extend(["--num_workers", str(checked_args.num_workers)])
        args_parse_list.extend(["--max_length", str(checked_args.max_len)])
        args_parse_list.extend(["--val_check_interval", str(checked_args.val_check_interval)])
        args_parse_list.extend(["--batch_size", str(checked_args.train_batchsize)])
        args_parse_list.extend(["--max_epochs", str(checked_args.max_epochs)])
        args_parse_list.extend(["--min_epochs", str(checked_args.min_epochs)])
        args_parse_list.extend(["--entity_multi_label"])
        args_parse_list.extend(["--relation_multi_label"])
        return args_parse_list

    @property
    def _inference_manager_type(self):
        return InferenceManagerIEStd

    def _parse_inference_args(self, args: GtsEngineArgs) -> List[str]:
        # args类型检查
        checked_args = TypeCheckedInferenceArgs(
            task_dir=Path(args.task_dir),
            pretrained_model_root=Path(args.pretrained_model_dir)
        )

        # 用户参数转化为实际参数
        args_parse_list: List[str] = []
        args_parse_list.extend(["--task_dir", str(checked_args.task_dir)])
        args_parse_list.extend(["--pretrained_model_root", str(checked_args.pretrained_model_root)])
        return args_parse_list
