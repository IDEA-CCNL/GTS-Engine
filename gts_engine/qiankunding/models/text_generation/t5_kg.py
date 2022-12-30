from genericpath import exists
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AdamW
import numpy as np

from ..text_classification.base_model import BaseModel
from gts_common.logs_utils import Logger
logger = Logger().get_log()


class T5KG(BaseModel):

    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)
        
        if isinstance(args, dict):
            args = argparse.Namespace(**args)

        self.config = T5Config.from_pretrained(args.pretrained_model)
        self.count = 0

        self.model = T5ForConditionalGeneration.from_pretrained(self.args.pretrained_model)
        self.model.resize_token_embeddings(new_num_tokens=len(self.tokenizer))

        self.init_model(args)
    
    def init_model(self, args):
        """
        init function.
        """
        pass

    def train_inputs(self, batch):
        inputs = {
            "input_ids": batch['input_ids'],
            "attention_mask": batch['attention_mask'],
            "labels": batch["labels"],
        }

        return inputs

    def training_step(self, batch, batch_idx):
        inputs = self.train_inputs(batch)
        outputs = self.model(**inputs)
        loss = outputs.loss

        self.log('train_loss', loss)

        return loss

    def training_epoch_end(self,training_step_outputs):

        if self.save_hf_model_file !='':
            ct=str(self.count)
            self.count+=1

    def validation_step(self, batch, batch_idx):

        inputs = self.train_inputs(batch)
        outputs = self.model.generate(
                    input_ids = inputs['input_ids'],
                    max_length = 32, 
                    )
        
        logits = outputs[:,1:]
        TP = 0
        total_pred = 0
        total_true = 0
        for i, j in zip(logits, batch["kpg_labels"]):
            
            predict_label = self.tokenizer.decode(i, skip_special_tokens=True)
            true_label = self.tokenizer.decode(j, skip_special_tokens=True)
            true_list = true_label.split('|')
            pred_list = predict_label.split('|')
            pred_list = list(set(pred_list))

            for pred in pred_list:
                if pred in true_list:
                    TP += 1

            total_pred += len(pred_list)
            total_true += len(true_list)


        precision = TP / total_pred
        recall = TP / total_true
        f1 = float(2 * (precision * recall)) / (precision + recall) if precision + recall > 0 else 0.0

        self.log('valid_precision', precision)
        self.log('valid_recall', recall)
        self.log('valid_f1', f1)
    
        return int(TP), int(total_pred), int(total_true)


    def validation_epoch_end(self, validation_step_outputs):
        TP = 0 
        total_pred = 0
        total_true = 0
        for x in validation_step_outputs:
            TP += x[0]
            total_pred += x[1]
            total_true += x[1]

        precision = TP / total_pred
        recall = TP / total_true
        f1 = float(2 * (precision * recall)) / (precision + recall) if precision + recall > 0 else 0.0

        self.log('valid_acc_epoch', f1, on_epoch=True, prog_bar=True)

        logger.info("TP = {}, precision = {}, recall = {}, f1 = {}".format(TP, precision, recall, f1))
        logger.info(f"valid_f1_epoch: {round(f1, 4)}")

    def predict_inputs(self, batch):
        inputs = {
            "input_ids": batch['input_ids'].cuda(),
            "attention_mask": batch['attention_mask'].cuda(),
            "labels": batch["labels"].cuda(),
        }

        return inputs


    def predict(self, batch):
        inputs = self.predict_inputs(batch)

        outputs = self.model.generate(
                    input_ids = inputs['input_ids'],
                    max_length=32
                    )
        
        logits=outputs[:,1:]

        predicts = []
        labels = []
        for i, j in zip(logits, batch["kpg_labels"]):

            predict_label = self.tokenizer.decode(i,skip_special_tokens=True)
            true_label = self.tokenizer.decode(j,skip_special_tokens=True)
            predicts.append(predict_label)
            labels.append(true_label)

        probs = []
        
        return logits, probs, predicts, labels

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

    
