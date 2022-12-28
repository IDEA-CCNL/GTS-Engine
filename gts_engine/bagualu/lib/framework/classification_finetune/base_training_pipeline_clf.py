from abc import abstractmethod, ABCMeta
from dataclasses import asdict
import datetime
from logging import Logger
import re
import os
import shutil
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.tokenization_utils import PreTrainedTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import Logger as PlLogger, TensorBoardLogger
from typing import List, Literal, Tuple, Dict, Union, Any
import torch
from pathlib import Path
from pydantic import DirectoryPath

from ...utils import LoggerManager
from ...components import TokenizerGenerator
from .prompt import StdPrompt
from ..base_training_pipeline import BaseTrainingPipeline
from ..consts import TRAINING_STAGE
from ...utils.json_processor import dump_json_list, load_json_list, dump_json, load_json
from ...utils.path import get_file_size
from ...components.metrics.clf_evaluation import get_confusion_matrix, get_classification_report
from ...utils.statistics import interval_mean, acc
from .base_arguments_clf import BaseTrainingArgumentsClf
from .base_data_module_clf import BaseDataModuleClf
from .base_lightnings_clf import BaseTrainingLightningClf
from .consts import DevOutput, LabeledSample, InfSampleProto, PredictionResult, TrainingSettings
from .data_reader_clf import DataReaderClf

class BaseTrainingPipelineClf(BaseTrainingPipeline, metaclass=ABCMeta):
    
    _args: BaseTrainingArgumentsClf
    _mode_name: str = "classification_ft"
    
    #############################################################################################
    ## overwrite
    #############################################################################################
    
    def _before_training(self) -> None:
        self._logger = self._get_logger()
        self._logger.info(f"use saving path: {self._output_dir}")
        self._logger.info(f"log_file: {self._log_file}")
        self._logger.info("selecting pretrainde model...") 
        self._select_pretrained_model()
        self._logger.info("generate tokenizer...")
        self._tokenizer = self._generate_tokenizer()
        self._logger.info("loading prompt...")
        self._prompt = self._load_prompt()
        
    def _train(self) -> None:
        self._data_module = self._get_data_module()
        self._data_module.reparse_args()
        self._training_lightning = self._get_training_lightning()
        self._trainer = self._get_trainer()
        self._logger.info(f"used gpus: {self._trainer.gpus}")
        self._fit()
        
    def _after_training(self) -> None:
        self._logger.info("select model from checkpoints...")
        best_ckpt = self._select_best_model_from_ckpts()
        if self._args.debug:
            if self._args.test_data_path is None:
                self._logger.info(
                    "test_data_path is not passed, skip testing...")
            elif not self._args.test_data_path.exists():
                self._logger.info(f"test_data_path {self._args.test_data_path}"
                                  f" does not exist, skip testing...")
            else:
                self._logger.info("implement test on training model...")
                self._implement_test(best_ckpt)
        self._logger.info("generating inference model...")
        state_dict = self._load_ckpt(best_ckpt).get_model_state_dict()
        self._inference_lightning = self._get_inf_lightning()
        self._inference_lightning.load_model_from_state_dict(state_dict) 
        self._inference_lightning.model.eval()
        self._logger.info("generate prediction file...")
        self._generate_prediction_file()
        if self._args.test_data_path is None:
            self._logger.info(
                "test_data_path is not passed, skip testing...")
        elif not self._args.test_data_path.exists():
            self._logger.info(f"test_data_path {self._args.test_data_path}"
                              f" does not exist, skip testing...")
        else:
            self._log_test_acc_on_inf_model()
        self._logger.info("save model files...")
        torch.save(state_dict, os.path.join(self._output_dir, "finetune_pytorch.bin"))
        self._logger.info("export model to onnx...")
        self._export_onnx()
        self._logger.info("generate and export evaluation results...")
        self._save_eval_results()
        self._logger.info("cleaning checkpoints...")
        for ckpt in self._ckpt_file_list:
            os.remove(ckpt)
        if os.path.exists(lightning_logs_path := os.path.join(self._output_dir, "lightning_logs")):
            shutil.rmtree(lightning_logs_path)
        self._copy_output_files()
    
    #############################################################################################
    ## private
    #############################################################################################
    
    #############################################################################################
    ##################################### before training #######################################
    
    def _get_logger(self) -> Logger:
        now_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._task_name = f"{self._mode_name}_{self._args.dataset}_{now_date}"
        self._log_file = os.path.join(self._args.log_dir, self._task_name + ".log")
        LoggerManager.set_logger(self._args.logger, self._log_file)
        return LoggerManager.get_logger(self._args.logger)
       
    @property
    def _output_dir(self) -> DirectoryPath:
        """输出地址"""
        return self._args.ft_output_dir
    
    def _select_pretrained_model(self): # todo 优化可读性
        if self._args.selected_pretrained_model_dir is not None and os.path.exists(self._args.selected_pretrained_model_dir):
            pretrained_model_dir = self._args.selected_pretrained_model_dir
            self._logger.info(f"using selected pretrained model: {pretrained_model_dir}")
        else:
            tapt_output_dir = os.path.join(self._args.student_output_dir, "tapt_output")
            if os.path.exists(tapt_output_dir) and set(["config.json", "pytorch_model.bin", "vocab.txt"]) <= set(os.listdir(tapt_output_dir)):
                pretrained_model_dir = tapt_output_dir
                self._logger.info(f"using tapt output as pretrained model: {pretrained_model_dir}")
            else:
                label2id = DataReaderClf.read_label2id(self._args.label2id_path)
                if len(label2id) == 2:
                    model = "Erlangshen-MacBERT-110M-BinaryClassification-Chinese"
                else:
                    sample_list = list(DataReaderClf.read_unlabeled_sample(self._args.train_data_path))
                    sentence_len_list = [len(sample.text) for sample in sample_list]
                    if interval_mean(sentence_len_list) < 100:
                        model = "chinese-macbert-base"
                    else:
                        model = "ernie-1.0-base-zh"
                pretrained_model_dir = os.path.join(self._args.pretrained_model_root, model)
                self._logger.info(f"using auto selected pretrained model: {pretrained_model_dir}")
        self._args.pretrained_model_dir = Path(pretrained_model_dir)
        
    def _generate_tokenizer(self) -> PreTrainedTokenizer:
        if not os.path.exists(self._args.pretrained_model_dir):
            pretrained_model_dir, model_name = os.path.split(self._args.pretrained_model_dir)
            if model_name=="chinese-macbert-base":
                huggingface_model_name = "hfl/chinese-macbert-base"
            elif model_name=="ernie-1.0-base-zh":
                huggingface_model_name = "xiaoqin/ernie-1.0-base-zh"
            else:
                huggingface_model_name = "IDEA-CCNL/Erlangshen-MacBERT-110M-BinaryClassification-Chinese"
            cache_path = os.path.join(pretrained_model_dir, model_name, "cache")
            tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name, cache_dir=cache_path)
            model = AutoModelForMaskedLM.from_pretrained(huggingface_model_name, cache_dir=cache_path)
            model.save_pretrained(os.path.join(pretrained_model_dir, model_name))
            tokenizer.save_pretrained(os.path.join(pretrained_model_dir, model_name))
            shutil.rmtree(cache_path)
            self._logger.info("local pretrained model path does not exist, model %s is downloaded from huggingface." % huggingface_model_name)
        return TokenizerGenerator.generate_tokenizer(self._args.pretrained_model_dir)
    
    def _load_prompt(self):
        try:
            assert(self._args.label2id_path is not None)
        except:
            raise Exception("no label2id path is passed")
        return StdPrompt(self._args.prefix_prompt, self._args.label2id_path)
    
    #############################################################################################
    ##################################### training ###########################################
        
    def _get_lightning_loggers(self) -> List[PlLogger]:
        tb_logger = TensorBoardLogger(
            save_dir=str(self._args.log_dir),
            name="tensor_board_logs",
            version=self._task_name,
            max_queue=2, 
            flush_secs=5
        )
        return [tb_logger]
    
    #############################################################################################
    ##################################### after training #############################################
    
    def _select_best_model_from_ckpts(self) -> str:
        """从保存的多个checkpoints中选择在全样本验证数据中表现最好的，返回checkpoint路径"""
        ckpt_file_list = self._ckpt_file_list
        if len(ckpt_file_list) == 0:
            raise Exception("no checkpoint is saved")
        if len(ckpt_file_list) == 1:
            return ckpt_file_list[0]
        acc_ckpt_tuple_list: List[Tuple[float, str]] = []
        for ckpt in ckpt_file_list:
            dev_output = self._dev_from_ckpt(ckpt, TRAINING_STAGE.VALIDATION)
            acc_ckpt_tuple_list.append((dev_output.dev_acc, ckpt))
        best_ckpt_tuple = max(acc_ckpt_tuple_list, key=lambda tp: tp[0])
        self._logger.info(f"best checkpoint dev acc: {best_ckpt_tuple[0]:.4f}")
        return best_ckpt_tuple[1]
    
    @property
    def _ckpt_file_list(self) -> List[str]:
        file_list = os.listdir(self._output_dir)
        ckpt_file_list = [os.path.join(self._output_dir, file) for file in file_list if re.match(r"^.*\.ckpt$", file) is not None]
        return ckpt_file_list
    
    def _dev_from_ckpt(self, checkpoint_path: str, stage: Literal[TRAINING_STAGE.TEST, TRAINING_STAGE.VALIDATION]) -> DevOutput:
        """输出对应checkpoint的验证/测试结果"""
        model = self._load_ckpt(checkpoint_path)
        data_loader = self._data_module.test_dataloader(stage=TRAINING_STAGE.TEST) if stage == TRAINING_STAGE.TEST else self._data_module.val_dataloader(stage=TRAINING_STAGE.VALIDATION, load_ratio=self._args.load_data_ratio)
        eval_output = self._trainer.validate(
            model=model, 
            dataloaders = data_loader,
            verbose=False
        )
        dev_acc = eval_output[0]["dev_acc"]
        dev_loss = eval_output[0]["dev_loss"]
        return DevOutput(dev_loss=dev_loss, dev_acc=dev_acc)
    
    def _load_ckpt(self, checkpoint_path: str) -> BaseTrainingLightningClf:
        return type(self._training_lightning).load_from_checkpoint(
                checkpoint_path,
                args=self._args,
                class_num=len(self._prompt.label2token),
                sample_num=self._data_module.train_sample_num
            )
    
    def _implement_test(self, ckpt: str) -> None:
        dev_output = self._dev_from_ckpt(ckpt, TRAINING_STAGE.TEST)
        self._logger.info(f"training model test acc: {dev_output.dev_acc:.4f}")

    def _generate_prediction_file(self):
        prediction_trainer = Trainer(
            accelerator="gpu",
            devices=1,
            default_root_dir=str(self._output_dir),
            enable_progress_bar=False,
            auto_select_gpus=True
        )
        if self._args.dev_data_path is None:
            self._logger.info(
                "dev_data_path is not passed, skip predicting on dev data...")
        elif not self._args.dev_data_path.exists():
            self._logger.info(f"dev_data_path {self._args.dev_data_path} does "
                              f"not exist, skip predicting on dev data...")
        else:
            self._logger.info("predicting on dev data..")
            dev_prediction_output = prediction_trainer.predict(
                model=self._inference_lightning,
                dataloaders=self._data_module.val_dataloader(stage=TRAINING_STAGE.INFERENCE)
            )
            dev_results = self._generate_prediction_results(dev_prediction_output, self._data_module.dev_sample_list) # type: ignore
            dump_json_list(dev_results, os.path.join(self._output_dir, "test_prediction_results.json"))
        if self._args.test_data_path is None:
            self._logger.info(
                "test_data_path is not passed, skip predicting on test data...")
        elif not self._args.test_data_path.exists():
            self._logger.info(f"test_data_path {self._args.test_data_path} does "
                              f"not exist, skip predicting on test data...")
        else:
            self._logger.info("predicting on test data..")
            test_prediction_output = prediction_trainer.predict(
                model=self._inference_lightning,
                dataloaders=self._data_module.test_dataloader(stage=TRAINING_STAGE.INFERENCE)
            )
            test_results = self._generate_prediction_results(test_prediction_output, self._data_module.test_sample_list) # type: ignore
            dump_json_list(test_results, os.path.join(self._output_dir, "offline_test_prediction_results.json"))
        if self._args.online_test_data_path is None:
            self._logger.info(
                "online_test_data_path is not passed, skip predicting on online test data...")
        elif not self._args.online_test_data_path.exists():
            self._logger.info(f"online_test_data_path {self._args.online_test_data_path} does "
                              f"not exist, skip predicting on online test data...")
        else:
            self._logger.info("predicting on online test data..")
            online_test_prediction_output = prediction_trainer.predict(
                model=self._inference_lightning,
                dataloaders=self._data_module.online_test_dataloader()
            )
            online_test_results = self._generate_prediction_results(online_test_prediction_output, self._data_module.online_test_sample_list) # type: ignore
            dump_json_list(online_test_results, os.path.join(self._output_dir, "online_test_prediction_results.json"))

    def _generate_prediction_results(self, prediction_output: Dict[int, str], sample_list: Union[List[InfSampleProto], List[LabeledSample]]) -> List[PredictionResult]:
        assert len(prediction_output) == len(sample_list)
        y_true = []
        y_pred = []
        result_list = []
        id2label = {key: val.label for key, val in self._prompt.id2label.items()}
        label2id = {val: key for key, val in id2label.items()}
        for idx, sample in enumerate(sample_list):
            prediction = prediction_output[idx]
            content = sample.text
            true_label = self._prompt.id2label[sample.label_id_clf].label if hasattr(sample, "label_id_clf") else None # type: ignore
            result_list.append(PredictionResult(
                id=sample.id,
                content=content,
                label=true_label,
                predict=prediction
            ))
            if hasattr(sample, "label_id_clf"):
                y_true.append(sample.label_id_clf)  # type: ignore
            y_pred.append(label2id[prediction])
        if len(y_true)>0:
            clf_report = get_classification_report(y_true, y_pred, id2label)
            accuracy = clf_report.pop('accuracy')
            self._logger.info(f"acc is: {accuracy:.4f}")
        return result_list
            
    def _export_onnx(self) -> None:
        batch_size = 2
        seq_len = 64
        dummy_input_ids = torch.ones([batch_size, seq_len],
                                     dtype=torch.long).cuda(0)
        dummy_input_mask = torch.ones([batch_size, seq_len],
                                      dtype=torch.long).cuda(0)
        dummy_input_seg = torch.ones([batch_size, seq_len],
                                     dtype=torch.long).cuda(0)
        input_names = ['input_ids', 'input_mask', 'input_seg']
        output_names = ['pred_labels', 'probs']
        dynamic_axes = {
            'input_ids': {
                0: 'batch_size',
                1: 'seq_length'
            },
            'input_mask': {
                0: 'batch_size',
                1: 'seq_length'
            },
            'input_seg': {
                0: 'batch_size',
                1: 'seq_length'
            },
            'pred_labels': {
                0: 'batch_size'
            },
            'probs': {
                0: 'batch_size',
            }
        }
        self._inference_lightning.model.eval()
        torch.onnx.export(
            self._inference_lightning.model.cuda(0), 
            (dummy_input_ids, dummy_input_mask, dummy_input_seg),
            os.path.join(self._output_dir, "model.onnx"),
            verbose=False,
            opset_version=12,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes)
        self._logger.info('Save ONNX model to {} successfully!'.format(os.path.join(self._output_dir, "model.onnx")))

    def _save_eval_results(self) -> None:
        settings = TrainingSettings(
            learning_rate=self._args.learning_rate,
            label_guided_rate=self._args.label_guided_rate,
            prefix_prompt=self._args.prefix_prompt,
            training_label_prompt=self._args.training_label_prompt,
            inference_label_prompt=self._args.inference_label_prompt,
            batch_size=self._args.train_batch_size,
            epoch=self._args.epoch,
            warmup_epoch=self._args.warm_up_epoch,
            decay_epoch=self._args.decay_epoch,
            dropout_rate=0.1
        )
        model_size = get_file_size(os.path.join(self._output_dir, "model.onnx"), "mb")
        if os.path.exists(os.path.join(self._output_dir, "test_prediction_results.json")):
            y_true, y_pred, id2label = self._load_test_results(os.path.join(self._output_dir, "test_prediction_results.json"))
            confusion_matrix = get_confusion_matrix(y_true, y_pred, id2label)
            clf_report = get_classification_report(y_true, y_pred, id2label)
            accuracy = clf_report.pop('accuracy')
            macro_avg = clf_report.pop("macro avg")
            weighted_avg = clf_report.pop("weighted avg")
            eval_results = {
                "global_indicator": {
                    "accuracy": accuracy,
                    "macro_avg": macro_avg,
                    "weighted_avg": weighted_avg
                },
                "settings": asdict(settings),
                "confusion_matrix": asdict(confusion_matrix),
                "label_result": clf_report,
                "model_size": model_size
            }
        else:
            eval_results = {
                "global_indicator": {
                    "accuracy": 0,
                    "macro_avg": 0,
                    "weighted_avg": 0
                },
                "settings": asdict(settings),
                "confusion_matrix": None,
                "label_result": None,
                "model_size": model_size
            }
        dump_json(eval_results, os.path.join(self._output_dir, "result.json"), indent=2)
        
    def _load_test_results(self, path: str):
        test_results_list = list(load_json_list(path, PredictionResult))
        id2label = {key: val.label for key, val in self._prompt.id2label.items()}
        label2id = {val: key for key, val in id2label.items()}
        y_true = [label2id[result.label] for result in test_results_list] # type: ignore
        y_pred = [label2id[result.predict] for result in test_results_list]
        return y_true, y_pred, id2label
    
    def _log_test_acc_on_inf_model(self) -> None:
        y_true, y_pred, _ = self._load_test_results(os.path.join(self._output_dir, "offline_test_prediction_results.json"))
        acc_ = acc(y_true, y_pred)
        self._logger.info(f"inference model test acc: {acc_:.4f}")
    
    def _copy_output_files(self) -> None:
        shutil.copy(os.path.join(self._args.pretrained_model_dir, "vocab.txt"), os.path.join(self._output_dir, "vocab.txt"))
        shutil.copy(os.path.join(self._args.pretrained_model_dir, "config.json"), os.path.join(self._output_dir, "config.json"))
        shutil.copy(self._log_file, os.path.join(self._output_dir, "training_log.log"))
        shutil.copy(os.path.join(self._output_dir, "result.json"), os.path.join(self._args.student_output_dir, "result.json"))
        shutil.copy(os.path.join(self._output_dir, "model.onnx"), os.path.join(self._args.student_output_dir, "model.onnx"))
        shutil.copy(os.path.join(self._output_dir, "vocab.txt"), os.path.join(self._args.student_output_dir, "vocab.txt"))
        shutil.copy(os.path.join(self._output_dir, "config.json"), os.path.join(self._args.student_output_dir, "config.json"))
        if os.path.exists(os.path.join(self._output_dir, "test_prediction_results.json")):
            shutil.copy(os.path.join(self._output_dir, "test_prediction_results.json"), os.path.join(self._args.student_output_dir, "test_prediction_results.json"))
    
    
    #############################################################################################
    ## abstract 
    #############################################################################################
    
    @abstractmethod
    def _get_data_module(self) -> BaseDataModuleClf: 
        """根据模式需要处理和加载数据"""
        
    @abstractmethod
    def _get_training_lightning(self) -> BaseTrainingLightningClf: 
        """根据模式需要加载训练模型"""
        
    @abstractmethod
    def _get_trainer(self) -> Trainer:
        """根据模式需要定义和加载Lightning Trainer"""
        
    @abstractmethod
    def _fit(self) -> None:
        """根据模式需要定义和实施训练"""
        
    @abstractmethod
    def _get_inf_lightning(self) -> BaseTrainingLightningClf:
        """根据模式需要加载推理模型"""