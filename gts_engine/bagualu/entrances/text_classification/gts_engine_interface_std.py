from typing import List, Optional, Dict
from pydantic import BaseModel, DirectoryPath, FilePath
from pathlib import Path

from ...lib.framework.base_gts_engine_interface import TRAIN_MODE, BaseGtsEngineInterface, GtsEngineArgs
from ...lib.utils.json_processor import dump_json, load_json

from .training_pipeline_std import TrainingPipelineClfStd
from .inference_manager_std import InferenceManagerClfStd


class TypeCheckedTrainArgs(BaseModel):
    """GTS-Engine相关参数进行runtime类型检查与转换"""
    task_dir: DirectoryPath
    pretrained_model_dir: DirectoryPath
    data_dir: DirectoryPath
    save_path: DirectoryPath
    train_data_path: FilePath
    valid_data_path: FilePath
    test_data_path: Optional[FilePath]
    label_data_path: Optional[FilePath]
    gpus: int
    train_mode: TRAIN_MODE
    seed: int
    num_workers: int
    train_batchsize: int
    valid_batchsize: int
    test_batchsize: int
    max_length: int
    learning_rate: float
    
class TypeCheckedInfArgs(BaseModel):
    model_save_dir: DirectoryPath
    label2id_path: FilePath


class GtsEngineInterfaceClfStd(BaseGtsEngineInterface):

    @property
    def _training_pipeline_type(self):
        return TrainingPipelineClfStd

    def _parse_training_args(self, args: GtsEngineArgs) -> List[str]:
        # args类型检查
        type_checked_args = TypeCheckedTrainArgs(
            task_dir=Path(args.task_dir),
            pretrained_model_dir=Path(args.pretrained_model_dir),
            data_dir=Path(args.data_dir),
            save_path=Path(args.save_path),
            train_data_path=args.train_data_path,
            valid_data_path=args.valid_data_path,
            test_data_path=args.test_data_path,
            # 将在prepare_training()中通过label_data生成label2id.json
            label_data_path=self.__get_label2id_path(args),
            gpus=args.gpus,
            train_mode=TRAIN_MODE(args.train_mode),
            seed=args.seed,
            num_workers=args.num_workers,
            train_batchsize=args.train_batchsize,
            valid_batchsize=args.valid_batchsize,
            test_batchsize=args.test_batchsize,
            max_length=args.max_len,
            learning_rate=args.lr,
        )
        args_parse_list: List[str] = []
        args_parse_list.extend(
            ["--gts_input_path", str(type_checked_args.task_dir)])
        args_parse_list.extend(
            ["--gts_pretrained_model_path", str(type_checked_args.pretrained_model_dir)])
        args_parse_list.extend(
            ["--gts_output_dir", str(type_checked_args.save_path)])
        args_parse_list.extend(["--gts_train_level", "1"])
        args_parse_list.extend(["--gpu_num", str(type_checked_args.gpus)])
        args_parse_list.extend(["--run_mode", "online"])
        args_parse_list.extend(
            ["--train_data_path", str(type_checked_args.train_data_path)])
        args_parse_list.extend(
            ["--dev_data_path", str(type_checked_args.valid_data_path)])
        args_parse_list.extend(
            ["--aug_eda_path", str(self.__get_eda_cache_path(args))])
        args_parse_list.extend(
            ["--num_workers", str(type_checked_args.num_workers)])
        args_parse_list.extend(
            ["--train_batchsize", str(type_checked_args.train_batchsize)])
        args_parse_list.extend(
            ["--valid_batchsize", str(type_checked_args.valid_batchsize)])
        args_parse_list.extend(
            ["--test_batchsize", str(type_checked_args.test_batchsize)])
        args_parse_list.extend(
            ["--max_length", str(type_checked_args.max_length)])
        args_parse_list.extend(
            ["--learning_rate", str(type_checked_args.learning_rate)])
        if type_checked_args.test_data_path is not None:
            args_parse_list.extend(
                ["--test_data_path", str(type_checked_args.test_data_path)])
        try:
            assert type_checked_args.label_data_path is not None
            args_parse_list.extend(
                ["--label2id_path", str(type_checked_args.label_data_path)])
        except:
            raise Exception(
                "you should pass label_data file in classification task")
        args_parse_list.extend(
            ["--log_dir", str(type_checked_args.task_dir / "logs")])
        return args_parse_list

    def prepare_training(self, args: GtsEngineArgs) -> None:
        # 将label_data数据转为label2id格式
        assert args.label_data_path is not None
        label_data = load_json(args.label_data_path,
                               type_=Dict[str, List[str]])
        label2id = {
            label: {"id": idx, "label_desc_zh": label}
            for idx, label in enumerate(label_data["labels"])
        }
        dump_json(label2id, self.__get_label2id_path(args))

    @property
    def _inference_manager_type(self):
        return InferenceManagerClfStd

    def _parse_inference_args(self, args: GtsEngineArgs) -> List[str]:
        type_checked_args = TypeCheckedInfArgs(
            label2id_path=self.__get_label2id_path(args),
            model_save_dir=Path(args.task_dir) / "outputs" /
            "student_output" / "finetune_output"
        )
        args_parse_list: List[str] = []
        args_parse_list.extend(
            ["--model_save_dir", str(type_checked_args.model_save_dir)])
        args_parse_list.extend(
            ["--label2id_path", str(type_checked_args.label2id_path)])
        return args_parse_list

    def __get_label2id_path(self, args: GtsEngineArgs) -> FilePath:
        return Path(args.task_dir) / f"{args.train_data.split('.')[0]}_label2id.json"

    def __get_eda_cache_path(self, args: GtsEngineArgs) -> FilePath:
        return Path(args.task_dir) / f"{args.train_data.split('.')[0]}_eda_augment.json"
