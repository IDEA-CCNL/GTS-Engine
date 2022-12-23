from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelSummary, ModelCheckpoint
from typing import List, Literal, Tuple, Dict, Union
import torch
import os
import shutil
import numpy as np

from ...lib.framework.classification_finetune import BaseTrainingPipelineClf
from ...lib.framework.classification_finetune.consts import InferenceManagerInputSample
from ...lib.framework.consts import TRAINING_STAGE
from ...lib.components.lightning_callbacks.adaptive_val_intervals import AdaptiveValIntervalTrainLoss, ADAPTIVE_VAL_INTERVAL_MODE, AdaptiveValIntervalFixed
from ...lib.components.knn_tools import BasicDatastore, grid_search_for_hyper, get_datastore

from ...arguments.text_classification.arguments_std import TrainingArgumentsClfStd
from ...dataloaders.text_classification.data_module_std import DataModuleClfStd
from ...models.text_classification.lightnings_std import TrainLightningClfStd, PredictLightningClfStd

class TrainingPipelineClfStd(BaseTrainingPipelineClf):
    
    _args: TrainingArgumentsClfStd
    _mode_name: str = "ft_std"
    
    def _get_data_module(self) -> DataModuleClfStd:
        return DataModuleClfStd(self._args, self._prompt, self._tokenizer)
    
    def _get_training_lightning(self) -> TrainLightningClfStd:
        return TrainLightningClfStd(self._args, self._data_module.class_num, self._data_module.train_sample_num)
    
    def _get_trainer(self) -> Trainer:
        callbacks = self.__get_callbacks(self._data_module.dev_sample_num)
        return Trainer(
            accelerator="gpu", 
            devices=self._args.gpu_num, 
            callbacks=callbacks,
            max_epochs=self._args.epoch,
            gradient_clip_val=self._args.clip_norm,
            default_root_dir=str(self._output_dir),
            enable_progress_bar=False,
            auto_select_gpus=True,
            strategy="dp",
            logger=self._get_lightning_loggers(),
            precision=self._args.precision,
        )
        
    def _fit(self) -> None:
        seed_everything(self._args.seed)
        self._logger.info('-' * 30 + 'Args' + '-' * 30)
        vars_str = ''
        for k, v in vars(self._args).items():
            vars_str += str(k)+':'+str(v)+'\t'
        self._logger.info(vars_str)
        self._logger.info('\n' + '-' * 64)

        self._trainer.fit(
            model=self._training_lightning, 
            train_dataloaders=self._data_module.train_dataloader(load_ratio=self._args.load_data_ratio),
            val_dataloaders=self._data_module.val_dataloader(
                stage=TRAINING_STAGE.VALIDATION, 
                load_ratio=self._args.load_data_ratio, 
                resample_thres=self._args.dev_resample_thres),
        )
        
    def _get_inf_lightning(self, use_knn=False, datastore=None, best_hyper=None) -> PredictLightningClfStd:
        if use_knn:
            return PredictLightningClfStd(self._prompt, self._args, self._tokenizer, datastore=datastore, best_hyper=best_hyper)
        else:
            return PredictLightningClfStd(self._prompt, self._args, self._tokenizer)
    
    def __get_callbacks(self, dev_num: int) -> List[Callback]:
        model_summary = ModelSummary(max_depth=2)
        if self._args.validation_mode == ADAPTIVE_VAL_INTERVAL_MODE.ADAPTIVE and self._data_module.class_num > 2: 
            adaptive_test_interval_cls = AdaptiveValIntervalTrainLoss
        else:
            adaptive_test_interval_cls = AdaptiveValIntervalFixed
        adaptive_test_interval = adaptive_test_interval_cls(
            int(self._data_module.train_sample_num * self._args.load_data_ratio),
            self._args.train_batch_size,
            self._args.logger
        )
        if dev_num == 0: # 无验证数据时，保存最终模型
            model_checkpoint = ModelCheckpoint(
                monitor="step",
                mode="max", 
                save_top_k=1,
                save_on_train_epoch_end=True, 
                verbose=True,
                dirpath=self._output_dir
            )
        elif dev_num <= self._args.dev_resample_thres: # 验证数据较少时，保存dev_loss最小模型
            model_checkpoint = ModelCheckpoint(
                monitor="dev_acc",
                mode="max", 
                save_top_k=1,
                save_on_train_epoch_end=False, 
                verbose=True,
                dirpath=self._output_dir
            )
        else: # 验证数据较多时，重采样验证数据，保存dev_loss最小的五个模型，训练完成后再用全样本筛选出最优模型
            self._logger.info("dev sample is too large, resample dev data...")
            model_checkpoint = ModelCheckpoint(
                monitor="dev_acc",
                mode="max", 
                save_top_k=5,
                save_on_train_epoch_end=False, 
                verbose=True,
                dirpath=self._output_dir
            )
        return [
            model_summary,
            adaptive_test_interval,
            model_checkpoint
        ]

    def _after_training(self) -> None:
        # select best model from ckpts
        self._logger.info("select model from checkpoints...")
        best_ckpt = self._select_best_model_from_ckpts()
        # The best ckpt model to make predictions on test_public, and outputs the accuracy
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
        # generating inference model, test predicting, and save onnx model
        self._logger.info("generating inference model...")
        state_dict = self._load_ckpt(best_ckpt).get_model_state_dict()
        self._inference_lightning = self._get_inf_lightning()
        self._inference_lightning.load_model_from_state_dict(state_dict)
        self._inference_lightning.model.eval()
        
        if self._args.use_knn:
            # Create a knn datastore with train data
            self._logger.info("generating knn datastore...")
            datastore = get_datastore(model=self._inference_lightning.model.cuda(), 
                                        data_model=self._get_data_module().raw_train_dataloader())
            best_hyper,_,_ = grid_search_for_hyper(datastore, 
                                                    self._get_data_module().val_dataloader(stage=TRAINING_STAGE.VALIDATION), 
                                                    self._inference_lightning.model.cuda())
            self._logger.info("best_hyper of knn is {}, base dev dataset".format(best_hyper))
            
            # construct knn_based inference model
            self._inference_lightning = self._get_inf_lightning(use_knn=self._args.use_knn, datastore=datastore, best_hyper=best_hyper)
            self._inference_lightning.load_model_from_state_dict(state_dict)
            self._inference_lightning.model.eval()
        self._logger.info("generate prediction file...")
        self._generate_prediction_file()
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