from pydantic import FilePath
from transformers.tokenization_utils import PreTrainedTokenizer
import os
from abc import ABCMeta, abstractmethod
from pytorch_lightning import Trainer
import torch
import shutil
from torch.utils.data import DataLoader

from ..base_inference_manager import BaseInferenceManager
from .base_arguments_clf import BaseInferenceArgumentsClf
from .base_lightnings_clf import BaseInferenceLightningClf
from .base_dataset_clf import BaseDatasetClf
from .consts import InferenceManagerInputSampleList, InferenceManagerOutput
from ..mixin import OptionalLoggerMixin
from ...components import TokenizerGenerator
from .prompt import StdPrompt

class BaseInferenceManagerClf(BaseInferenceManager, OptionalLoggerMixin):
    
    _args: BaseInferenceArgumentsClf
    
    def prepare_inference(self) -> None:
        self.info(f"loading model...")
        self.info(f"generate tokenizer...")
        self._tokenizer = self._generate_tokenizer()
        self.info(f"loading prompt...")
        self._prompt = self._load_prompt()
        self.info(f"loading model...")
        self._inf_lightning = self._get_inf_lightning()
        self._inf_lightning.load_model_from_state_dict(torch.load(self._args.model_state_dict_file_path))
        self.info(f"init prediction lightning trainer")
        self._trainer =  Trainer(
            accelerator="gpu",
            devices=1,
            default_root_dir=str(self._args.model_save_dir / "tmp"),
            enable_progress_bar=False,
            auto_select_gpus=True
        )
        
    def inference(self, sample: InferenceManagerInputSampleList) -> InferenceManagerOutput:
        self.info(f"processing data...")
        dataset = self._get_dataset(sample)
        dataloader = DataLoader(dataset, batch_size=self._args.batch_size, num_workers=6)
        self.info(f"predicting on data...")
        inf_output: InferenceManagerOutput = self._trainer.predict( # type: ignore
            model=self._inf_lightning,
            dataloaders=dataloader
        )
        shutil.rmtree(str(self._args.model_save_dir / "tmp"))
        return inf_output
        
    
    def _generate_tokenizer(self) -> PreTrainedTokenizer:
        return TokenizerGenerator.generate_tokenizer(self._args.model_save_dir)
    
    def _load_prompt(self):
        return StdPrompt(self._args.prefix_prompt, self._args.label2id_path)
    
    @abstractmethod
    def _get_inf_lightning(self) -> BaseInferenceLightningClf: ...
    
    @abstractmethod
    def _get_dataset(self, sample: InferenceManagerInputSampleList) -> BaseDatasetClf: ...
    
    
    
    