from pytorch_lightning import LightningModule
import torch
from typing import Dict, Any
from logging import Logger
from abc import ABCMeta, abstractmethod
from typing import List

from .consts import InfBatch, InferenceManagerOutput

class BaseTrainingLightningClf(LightningModule):
    
    _model: torch.nn.Module
    _logger: Logger
    
    def get_model_state_dict(self):
        return self._model.state_dict()
    
    def save_model(self, path: str):
        torch.save(self.get_model_state_dict(), f=path)
        
    def load_model_from_state_dict(self, state_dict: Dict[str, Any]):
        self._model.load_state_dict(state_dict, strict=False) # type: ignore
        
    @property
    def model(self):
        return self._model
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        try:
            dev_acc: float = self.trainer.logged_metrics["dev_acc"]
            self._logger.info(f"save checkpoint. dev acc: {dev_acc:.4f}")
        except:
            pass
        return super().on_save_checkpoint(checkpoint)
    

class BaseInferenceLightningClf(LightningModule):
    
    _model: torch.nn.Module
    
    def load_model_from_state_dict(self, state_dict: Dict[str, Any]):
        self._model.load_state_dict(state_dict, strict=False) # type: ignore
        
    @abstractmethod
    def predict_step(self, batch: InfBatch, batch_idx: int, dataloader_idx: int = 0) -> InferenceManagerOutput:
        ...
        
    def on_predict_epoch_end(self, results: List[List[InferenceManagerOutput]]) -> None:
        res = InferenceManagerOutput(predictions=[], probabilities=[])
        for inf_output in results[0]:
            res["predictions"].extend(inf_output["predictions"])
            res["probabilities"].extend(inf_output["probabilities"])
        results[0] = res # type: ignore 