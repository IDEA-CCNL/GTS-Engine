#encodoing=utf8
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from transformers import AutoConfig, AutoModelForMaskedLM, MegatronBertForMaskedLM, MegatronBertConfig
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AdamW,Adafactor
from .base_model import BaseModel, Pooler

from ...utils.detect_gpu_memory import detect_gpu_memory
from ...utils import globalvar as globalvar
from gts_common.logs_utils import Logger

logger = Logger().get_log()


class taskModel(nn.Module):
    def __init__(self, pre_train_dir: str, tokenizer, nlabels, config):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pre_train_dir)
        if "1.3B" in pre_train_dir:
            # v100
            logger.info(globalvar.get_value("gpu_type"))
            if globalvar.get_value("gpu_type") == "low_gpu":
                self.config.gradient_checkpointing = True
                self.bert_encoder = MegatronBertForMaskedLM.from_pretrained(pre_train_dir, config=self.config)
                logger.info("使用gradient_checkpointing！")
            elif globalvar.get_value("gpu_type") == "mid_gpu":
                self.config.gradient_checkpointing = True
                self.bert_encoder = MegatronBertForMaskedLM.from_pretrained(pre_train_dir, config=self.config)
                logger.info("使用gradient_checkpointing！")
            elif globalvar.get_value("gpu_type") == "high_gpu":
                self.bert_encoder = MegatronBertForMaskedLM.from_pretrained(pre_train_dir)
            else:
                self.bert_encoder = MegatronBertForMaskedLM.from_pretrained(pre_train_dir)
        else:
            self.bert_encoder = AutoModelForMaskedLM.from_pretrained(pre_train_dir)
        self.bert_encoder.resize_token_embeddings(new_num_tokens=len(tokenizer))
        
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

        self.dropout = nn.Dropout(0.1)
        self.nlabels = nlabels
        self.linear_classifier = nn.Linear(config.hidden_size, self.nlabels)

    def forward(self, input_ids, attention_mask, token_type_ids,position_ids=None, mlmlabels=None, clslabels=None, clslabels_mask=None, mlmlabels_mask=None):

        batch_size,seq_len=input_ids.shape
        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    token_type_ids=token_type_ids,
                                    labels=mlmlabels,
                                    output_hidden_states=True)  # (bsz, seq, dim)


        mlm_logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        cls_logits = hidden_states[:,0]
        cls_logits = self.dropout(cls_logits)

        logits = self.linear_classifier(cls_logits)

        
        return outputs.loss, logits, mlm_logits, hidden_states


class TCBert(BaseModel):

    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)
        
        if isinstance(args, dict):
            args = argparse.Namespace(**args)

        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.hidden_size = self.config.hidden_size

        self.save_hf_model_path = os.path.join(args.save_path,"hf_model/")
        self.save_hf_model_file = os.path.join(self.save_hf_model_path,"pytorch_model.bin")
        self.count = 0

        line = json.load(open(os.path.join(args.data_dir, args.label_data), 'r', encoding='utf8'))
        nlabels = len(line['labels'])

        self.model = taskModel(args.pretrained_model, self.tokenizer, nlabels, self.config)
        
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

        self.init_model(args)
    
    def init_model(self, args):
        """
        init function.
        """
        pass


    def train_inputs(self, batch):
        #  Filter reduntant information(for example: 'sentence') that will be passed to model.forward()
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids'],
        }
        return inputs 


    def training_step(self, batch, batch_idx):
        inputs = self.train_inputs(batch)
        labels = batch['labels']
        _, logits, mlm_logits, _ = self.model(**inputs)

        if labels is not None:
            loss = self.loss_fn(logits, labels.view(-1))

        ntotal = logits.size(0)
        ncorrect = (logits.argmax(dim=-1) == labels).long().sum()
        acc = ncorrect / ntotal

        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, prog_bar=True)

        return loss



    def training_epoch_end(self,training_step_outputs):

        if self.save_hf_model_file !='':
            ct=str(self.count)
            # save_path=self.save_hf_model_file.replace('.bin','-'+ct+'.bin')
            # torch.save(self.model.bert_encoder.state_dict(), f=save_path)
            logger.info('save the best model')
            self.count+=1


    def validation_step(self, batch, batch_idx):
        inputs = self.train_inputs(batch)
        
        labels = batch['labels']
        _, logits, mlm_logits, _ = self.model(**inputs)

        predict = logits.argmax(dim=-1).cpu().tolist()

        if labels is not None:
            loss = self.loss_fn(logits, labels.view(-1))

        ntotal = logits.size(0)
        
        ncorrect = int((logits.argmax(dim=-1) == batch['labels']).long().sum())
        acc = ncorrect / ntotal

        self.log('valid_loss', loss, on_step=True, prog_bar=True)
        self.log("valid_acc", acc, on_step=True, prog_bar=True)

        return int(ncorrect), int(ntotal)



    def validation_epoch_end(self, validation_step_outputs):
        gpu_memory, gpu_used_memory = detect_gpu_memory()
        if gpu_used_memory > globalvar.get_value("gpu_max_used_memory"):
            globalvar.set_value("gpu_max_used_memory", gpu_used_memory)

        ncorrect = 0
        ntotal = 0
        for x in validation_step_outputs:
            ncorrect += x[0]
            ntotal += x[1]

        self.log('valid_acc_epoch', ncorrect / ntotal, on_epoch=True, prog_bar=True)

        logger.info("ncorrect = {}, ntotal = {}".format(ncorrect, ntotal))
        logger.info(f"Validation Accuracy: {round(ncorrect / ntotal, 4)}")


    def predict_inputs(self, batch):
        #  Filter reduntant information(for example: 'sentence') that will be passed to model.forward()
        inputs = {
            'input_ids': batch['input_ids'].cuda(),
            'attention_mask': batch['attention_mask'].cuda(),
            'token_type_ids': batch['token_type_ids'].cuda(),
        }
        return inputs 

    def predict(self, batch):
        inputs = self.predict_inputs(batch)
        with torch.no_grad():
            loss, logits, mlm_logits, hidden_states = self.model(**inputs)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicts = torch.argmax(probs, dim=-1)

        probs = probs.detach().cpu().numpy()
        predicts = predicts.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()

        labels = None
        if batch["labels"] is not None:
            labels = batch["labels"].detach().cpu().numpy()

        sample_embeds = None
        
        return logits, probs, predicts, labels, sample_embeds

    def configure_optimizers(self):

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        paras = list(
            filter(lambda p: p[1].requires_grad, self.named_parameters()))
        paras = [{
            'params':
            [p for n, p in paras if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.1
        }, {
            'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]
        if globalvar.get_value("gpu_type") == "low_gpu":
            optimizer = Adafactor(paras, lr=self.hparams.lr,relative_step=False, warmup_init=False)
            logger.info("使用Adafactor!")
        else:
            optimizer = torch.optim.AdamW(paras, lr=self.hparams.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.total_step * 0.1),
            self.total_step)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]