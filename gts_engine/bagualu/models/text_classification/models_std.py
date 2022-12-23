import torch
from torch import Tensor, nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForMaskedLM
from typing import Optional, List, Protocol, Union
from transformers.modeling_outputs import MaskedLMOutput
from dataclasses import asdict
from transformers.tokenization_utils import PreTrainedTokenizer
from pathlib import Path
import numpy as np
from sklearn.metrics import pairwise_distances

from ...lib.framework.consts import BertInput
from ...lib.framework.classification_finetune.consts import PromptToken, TrainingModelOutput, InferenceModelOutput
from ...lib.framework.classification_finetune import StdPrompt
from ...lib.components.losses import MaxMultiLogits, LabelSmoothing
from ...lib.components.knn_tools import inference_with_knn

from ...arguments.text_classification.arguments_std import TrainingArgumentsClfStd


class TrainingModelClfStd(nn.Module):
    
    def __init__(self, args, pretrained_model_dir: Union[str, Path], class_num: int, last_layers: int = 1):
        super().__init__()
        self._config = BertConfig.from_pretrained(pretrained_model_dir)
        args.memory_optimization_setting
        if args.use_gradient_checkpointing=="True":
            self._config.gradient_checkpointing=True
            print("使用gradient_checkpointing！")
        self._bert_encoder: BertForMaskedLM = BertForMaskedLM.from_pretrained(pretrained_model_dir, config=self._config) # type: ignore
        self._last_layers = last_layers
        self._multiply = 1 if last_layers == 4 else 3
        
        self._max_logits = MaxMultiLogits(
            class_num=class_num,
            hidden_size=last_layers * self._config.hidden_size, 
            multiply=self._multiply
        )
        self._label_smoothing = LabelSmoothing()
        self._softmax = nn.Softmax(dim=-1)
        self._KL_criterion = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        bert_input: BertInput,
        sample_weight: Optional[Tensor] = None,
        label_id_clf: Optional[Tensor] = None,
        is_training: bool = True
    ) -> TrainingModelOutput:
        bert_output: MaskedLMOutput = self._bert_encoder.forward(**bert_input, output_hidden_states=True) # type: ignore
        loss_mlm = bert_output.loss
        logits = bert_output.logits
        hidden_states = bert_output.hidden_states
        assert loss_mlm is not None and logits is not None and hidden_states is not None
        x_layer_cls: List[Tensor] = [hidden_states[-x] for x in range(1, self._last_layers + 1)]
        input_hidden_states = torch.cat(x_layer_cls, dim=-1)
        loss_ce = torch.tensor(0.)
        kl_loss = torch.tensor(0.)
        if label_id_clf is not None:
            cls_logits = self._max_logits.forward(input_hidden_states)
            logits = cls_logits
            loss_cls = self._label_smoothing.forward(cls_logits, label_id_clf.view(-1))
            if sample_weight is not None:
                proced_weight = sample_weight.view([-1, 1])
                loss_cls = loss_cls * proced_weight
            loss_ce = loss_cls.float().mean()
        loss = loss_ce + loss_mlm if is_training else loss_ce
        return TrainingModelOutput(
            loss_total=loss,
            loss_ce=loss_ce,
            loss_mlm=loss_mlm,
            kl_loss=kl_loss,
            logits=logits,
            loss_rd=torch.tensor(0.0),
            loss_ctr=torch.tensor(0.0),
            loss_ner=torch.tensor(0.0)
        )


class InfModelArgsProto(Protocol):
    pretrained_model_dir: Path
    inference_label_prompt: str
    max_length: int

class InferenceModelClfStd(nn.Module):

    def __init__(
        self,
        prompt: StdPrompt,
        args: InfModelArgsProto,
        tokenizer: PreTrainedTokenizer,
        last_layers: int = 1,
        datastore=None,
        best_hyper=None,
    ):

        super().__init__()
        self._config = BertConfig.from_pretrained(args.pretrained_model_dir)
        self._bert_encoder: BertForMaskedLM = BertForMaskedLM(self._config)
        self._last_layers = last_layers
        self._multiply = 1 if last_layers == 4 else 3
        
        self._max_logits = MaxMultiLogits(
            class_num=len(prompt.label2token),
            hidden_size=last_layers * self._config.hidden_size, 
            multiply=self._multiply
        )
        self._softmax = torch.nn.Softmax(dim=-1)
        
        predict_prompt = args.inference_label_prompt
        prompt_ = prompt.prompt + predict_prompt
        prompt_tokens = tokenizer.tokenize(prompt_)
        encode_dict = tokenizer.encode_plus(prompt_)
        self._prompt_ids = torch.tensor(
            encode_dict['input_ids']).long().unsqueeze(0)
        self._prompt_mask = torch.tensor(encode_dict['attention_mask']).long().unsqueeze(0)
        self._prompt_seg = torch.tensor(encode_dict['token_type_ids']).long().unsqueeze(0)
        self._max_length = args.max_length

        self.datastore = datastore
        self.best_hyper = best_hyper
        self.sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]  # type: ignore

    def forward(self, input_ids: Tensor, input_mask: Tensor, input_seg: Tensor, prompt_gate=True):
        batch_len = input_seg.shape[0]
        device = self._bert_encoder.device
        if prompt_gate:
            # 无标注数据/onnx推理时，需拼接prompt
            prompt_ids = self._prompt_ids.repeat(batch_len, 1).to(device)
            prompt_mask = self._prompt_mask.repeat(batch_len, 1).to(device)
            prompt_seg = self._prompt_seg.repeat(batch_len, 1).to(device)

            input_ids = torch.cat((prompt_ids, input_ids.to(device)[:, 1:]), 1)[:, :self._max_length]
            input_ids[:,-1][(input_ids[:,-1]>0).nonzero()] = self.sep_id #拼接后的input_ids特殊处理，保证最末有效token是[SEP]
            input_seg = input_mask #句对模式，因此需修改句子的input_seg为1，此时等于input_mask
            input_mask = torch.cat((prompt_mask, input_mask.to(device)[:, 1:]), 1)[:, :self._max_length]
            input_seg = torch.cat((prompt_seg, input_seg.to(device)[:, 1:]), 1)[:, :self._max_length]
        with torch.no_grad():
            bert_output = self._bert_encoder.forward(
                input_ids=input_ids.to(device),
                attention_mask=input_mask.to(device),
                token_type_ids=input_seg.to(device),
                output_hidden_states=True
            )
            x_layer_cls = [bert_output.hidden_states[-x] for x in range(1, self._last_layers + 1)] # type: ignore
            input_hidden_states: Tensor = torch.cat(x_layer_cls, dim=-1) # type: ignore
            max_logits = self._max_logits.forward(input_hidden_states)
            probs = self._softmax.forward(max_logits)
            positions = torch.argmax(probs, dim=-1)

            # 最后一层的平均作为句向量表征
            embeds = bert_output.hidden_states[-1].mean(1)  # type: ignore
            if self.datastore is not None:
                # knn_inference 
                knn_prob = inference_with_knn(self.datastore, probs, embeds, self.best_hyper)  # type: ignore
                knn_prob = self._softmax.forward(knn_prob)
                probs = knn_prob
                positions = torch.argmax(probs, dim=-1)
            
        return InferenceModelOutput(
            positions=positions,
            probs=probs,
            embeds=embeds
        )