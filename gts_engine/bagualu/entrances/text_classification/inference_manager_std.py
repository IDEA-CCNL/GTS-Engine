from typing import List

from ...lib.framework.classification_finetune import BaseInferenceManagerClf
from ...lib.framework.classification_finetune.consts import InferenceManagerInputSampleList

from ...models.text_classification.lightnings_std import InferenceLightningClfStd
from ...dataloaders.text_classification.datasets_std import InfDatasetClfStd
from ...arguments.text_classification.arguments_std import InferenceArgumentsClfStd

class InferenceManagerClfStd(BaseInferenceManagerClf):
    
    _args: InferenceArgumentsClfStd

    def _get_inf_lightning(self) -> InferenceLightningClfStd:
        return InferenceLightningClfStd(
            self._prompt, 
            self._args,
            self._tokenizer
        )
    
    def _get_dataset(self, sample: List[InferenceManagerInputSampleList]) -> InfDatasetClfStd:
        return InfDatasetClfStd(
            sample, self._tokenizer, self._prompt, 
            self._args.inference_label_prompt,
            self._args.prefix_prompt,
            self._args.max_length
        )