from dataclasses import asdict
from typing import Sequence
import numpy as np
import torch

from ...lib.framework.classification_finetune import BaseDatasetClf
from ...lib.framework.classification_finetune.consts import LabeledSample, PreEncodedTrainSample, EncodedTrainSample, InfSampleProto, EncodedInfSample
from ...lib.framework.classification_finetune.mask_tools import wwm_masking
from ...lib.components.text_tools import segment_text

class TrainDatasetClfStd(BaseDatasetClf):
    
    def __init__(self, sample_list, tokenizer, prompt, training_label_prompt: str, label_guided_rate: float, max_length: int, wwm_mask_rate: float):
        self._training_label_prompt = training_label_prompt
        self._label_guided_rate = label_guided_rate
        self._max_length = max_length
        self._wwm_mask_rate = wwm_mask_rate
        super().__init__(sample_list, tokenizer, prompt)
    
    def _encode_before_iter(self, sample: LabeledSample, idx: int) -> PreEncodedTrainSample:
        words = segment_text(sample.text)
        inference_prompt = segment_text(self._prompt.prompt, self._tokenizer) + self._tokenizer.tokenize(self._training_label_prompt)
        prompt_t = self._prompt.prompt + sample.label
        training_prompt = segment_text(prompt_t, self._tokenizer)
        return PreEncodedTrainSample(**asdict(sample), words=words, inference_prompt=inference_prompt, training_prompt=training_prompt)
    
    def _encode_on_iter(self, sample: PreEncodedTrainSample, idx: int) -> EncodedTrainSample:
        rands = np.random.random(1)
        if rands[0] < self._label_guided_rate:
            prompt_tokens = sample.inference_prompt
        else:
            prompt_tokens = sample.training_prompt
        prompt_tokens.insert(0, "[CLS]")
        prompt_tokens.append("[SEP]")

        max_second_len = self._max_length - len(prompt_tokens) - 3
        sentence_tokens = sample.words[:max_second_len]
        sentence_tokens.append("[SEP]")

        new_tokens = prompt_tokens + sentence_tokens
        token_type_ids = [0] * len(prompt_tokens) + \
            [1] * (len(sentence_tokens))
        attention_mask = [1] * len(new_tokens)
        assert len(token_type_ids) == len(attention_mask)
        while len(token_type_ids) < self._max_length:
            token_type_ids.append(0)
            attention_mask.append(0)
        wwm_output = wwm_masking(
            new_tokens=new_tokens,
            index=idx,
            tokenizer=self._tokenizer,
            prompt_mode=self._prompt,
            max_length=self._max_length,
            mask_rate=self._wwm_mask_rate
        )
        source = wwm_output.input_ids
        target = wwm_output.mlm_labels
        attention_mask = wwm_output.input_mask
        mask_positions = wwm_output.masked_lm_positions
        assert len(source) == len(target) == len(token_type_ids)
        while len(mask_positions) <= 30:
            mask_positions.append(-100)
        end_token = ["[SEP]"]
        end_id_list = self._tokenizer.convert_tokens_to_ids(end_token)
        assert isinstance(end_id_list, list)
        end_id = end_id_list[0]
        seq_actual_len = len(source) - source[::-1].index(end_id)
        mask_positions = mask_positions[:30]
        return EncodedTrainSample(
            input_ids=torch.tensor(source).long(),
            input_seg=torch.tensor(token_type_ids).long(),
            input_mask=torch.tensor(attention_mask).long(),
            mask_positions=torch.tensor(mask_positions).long(),
            labels=torch.tensor(target).long(),
            label_id=torch.tensor(sample.label_id).long(),
            label_id_clf=torch.tensor(sample.label_id_clf).long(),
            id=sample.id,
            weight=torch.tensor(1).float(),
            soft_label=torch.tensor(sample.soft_label).float(),
            seq_len=seq_actual_len,
            my_id=idx
        )
        
class TestDatasetClfStd(BaseDatasetClf):
    
    def __init__(self, sample_list, tokenizer, prompt, inference_label_prompt: str, prefix_prompt: str, max_length: int):
        self._inference_label_prompt = inference_label_prompt
        self._prefix_prompt = prefix_prompt
        self._max_length = max_length
        super().__init__(sample_list, tokenizer, prompt)
    
    def _encode_before_iter(self, sample: LabeledSample, idx: int) -> EncodedTrainSample:
        mask_positions = []
        predict_prompt = self._inference_label_prompt
        prompt = self._prefix_prompt + predict_prompt
        encode_dict = self._tokenizer.encode_plus(prompt,
                                                 text_pair=sample.text,
                                                 truncation="only_second",
                                                 max_length=self._max_length,
                                                 pad_to_max_length=True)
        encode_sent = encode_dict["input_ids"]
        token_type_ids = encode_dict["token_type_ids"]
        attention_mask = encode_dict["attention_mask"]
        source, target = encode_sent[:], [-100] * len(encode_sent) # type: ignore

        while len(mask_positions) <= 30:
            mask_positions.append(-100)

        end_token = ["[SEP]"]
        end_id = self._tokenizer.convert_tokens_to_ids(end_token)[0]  # type: ignore
        seq_actual_len = len(source) - source[::-1].index(end_id)

        return EncodedTrainSample(
            input_ids=torch.tensor(source).long(),
            input_seg=torch.tensor(token_type_ids).long(),
            input_mask=torch.tensor(attention_mask).long(),
            labels=torch.tensor(target).long(),
            mask_positions=torch.tensor(mask_positions[:30]).long(),
            seq_len=seq_actual_len,
            label_id=torch.tensor(sample.label_id).long(),
            label_id_clf=torch.tensor(sample.label_id_clf).long(),
            id=sample.id,
            weight=torch.tensor(1).float(),
            soft_label=torch.tensor(sample.soft_label).float(),
            my_id=idx
        )
    
class InfDatasetClfStd(BaseDatasetClf):
    
    def __init__(self, sample_list, tokenizer, prompt, inference_label_prompt: str, prefix_prompt: str, max_length: int):
        self._inference_label_prompt = inference_label_prompt
        self._prefix_prompt = prefix_prompt
        self._max_length = max_length
        super().__init__(sample_list, tokenizer, prompt)
    
    def _encode_before_iter(self, sample: InfSampleProto, idx: int) -> EncodedInfSample:
        mask_positions = []
        predict_prompt = self._inference_label_prompt
        prompt = self._prefix_prompt + predict_prompt
        encode_dict = self._tokenizer.encode_plus(sample.text,
                                                 max_length=self._max_length,
                                                 pad_to_max_length=True)
        encode_sent = encode_dict["input_ids"]
        token_type_ids = encode_dict["token_type_ids"]
        attention_mask = encode_dict["attention_mask"]
        source, target = encode_sent[:], [-100] * len(encode_sent) # type: ignore

        while len(mask_positions) <= 30:
            mask_positions.append(-100)

        end_token = ["[SEP]"]
        end_id = self._tokenizer.convert_tokens_to_ids(end_token)[0]  # type: ignore
        seq_actual_len = len(source) - source[::-1].index(end_id)

        return EncodedInfSample(
            input_ids=torch.tensor(source).long(),
            input_seg=torch.tensor(token_type_ids).long(),
            input_mask=torch.tensor(attention_mask).long(),
            my_id=idx
        )