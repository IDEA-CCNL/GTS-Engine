#encodoing=utf8
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModelForMaskedLM, MegatronBertForMaskedLM, MegatronBertConfig
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AdamW,Adafactor
from .base_model import BaseModel, Pooler


import numpy as np

from ...utils.detect_gpu_memory import detect_gpu_memory
from ...utils import globalvar as globalvar


class taskModel(nn.Module):
    def __init__(self, pre_train_dir: str, tokenizer):
        super().__init__()
        self.yes_token = tokenizer.encode("是")[1]
        self.no_token = tokenizer.encode("非")[1]
        
        if "1.3B" in pre_train_dir:
            # v100
            print(globalvar.get_value("gpu_type"))
            if globalvar.get_value("gpu_type") == "low_gpu":
                self.config.gradient_checkpointing = True
                self.bert_encoder = MegatronBertForMaskedLM.from_pretrained(pre_train_dir, config=self.config)
                print("使用gradient_checkpointing！")
            elif globalvar.get_value("gpu_type") == "mid_gpu":
                self.config.gradient_checkpointing = True
                self.bert_encoder = MegatronBertForMaskedLM.from_pretrained(pre_train_dir, config=self.config)
                print("使用gradient_checkpointing！")
            elif globalvar.get_value("gpu_type") == "high_gpu":
                self.bert_encoder = MegatronBertForMaskedLM.from_pretrained(pre_train_dir)
            else:
                self.bert_encoder = MegatronBertForMaskedLM.from_pretrained(pre_train_dir)
        else:
            self.config = AutoConfig.from_pretrained(pre_train_dir)
            self.bert_encoder = AutoModelForMaskedLM.from_pretrained(pre_train_dir)
        self.bert_encoder.resize_token_embeddings(new_num_tokens=len(tokenizer))

        # self.cls_layer = nn.Linear(self.config.hidden_size, 1)
        
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, input_ids, attention_mask, token_type_ids,position_ids=None, mlmlabels=None, clslabels=None, clslabels_mask=None, mlmlabels_mask=None):

        batch_size,seq_len=input_ids.shape
        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    token_type_ids=token_type_ids,
                                    labels=mlmlabels,
                                    output_hidden_states=True)  # (bsz, seq, dim)
        mask_loss = outputs.loss
        mlm_logits = outputs.logits
        cls_logits = mlm_logits[:,:,self.yes_token].view(-1,seq_len)+clslabels_mask
        hidden_states = outputs.hidden_states[-1]

        if mlmlabels == None:
            return 0, mlm_logits, cls_logits
        else:
            cls_loss = self.loss_func(cls_logits,clslabels)
            all_loss = mask_loss+cls_loss
            # all_loss = mask_loss
            return all_loss, mlm_logits, cls_logits, hidden_states

class BertUnifiedMC(BaseModel):

    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)
        
        if isinstance(args, dict):
            args = argparse.Namespace(**args)

        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.hidden_size = self.config.hidden_size

        self.yes_token = self.tokenizer.encode("是")[1]
        self.no_token = self.tokenizer.encode("非")[1]
        self.sep_token = tokenizer.encode("[SEP]")[1]

        self.save_hf_model_path = os.path.join(args.save_path,"hf_model/")
        self.save_hf_model_file = os.path.join(self.save_hf_model_path,"pytorch_model.bin")
        self.count = 0

        self.model = taskModel(args.pretrained_model, self.tokenizer)
        
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
            "position_ids":batch['position_ids'],
            "mlmlabels": batch['mlmlabels'],
            "clslabels": batch['clslabels'],
            "clslabels_mask": batch['clslabels_mask'],
            "mlmlabels_mask": batch['mlmlabels_mask'],
        }
        # print("batch长度：",inputs["input_ids"].shape)
        return inputs 


    def training_step(self, batch, batch_idx):
        inputs = self.train_inputs(batch)
        loss, logits, cls_logits, _ = self.model(**inputs)
        mask_acc = self.comput_metrix(logits, batch['mlmlabels'], batch['mlmlabels_mask'])
        cls_acc, ncorrect, ntotal = self.comput_metrix(cls_logits, batch['clslabels'])

        self.log('train_loss', loss)
        self.log('train_mask_acc', mask_acc)
        self.log('train_cls_acc', cls_acc)

        return loss

    def training_epoch_end(self,training_step_outputs):

        if self.save_hf_model_file !='':
            ct=str(self.count)
            # save_path=self.save_hf_model_file.replace('.bin','-'+ct+'.bin')
            # torch.save(self.model.bert_encoder.state_dict(), f=save_path)
            print('save the best model')
            self.count+=1
    
    def validation_step(self, batch, batch_idx):
        inputs = self.train_inputs(batch)
        loss, logits, cls_logits, _ = self.model(**inputs)
        mask_acc = self.comput_metrix(logits, batch['mlmlabels'], batch['mlmlabels_mask'])
        cls_acc, ncorrect, ntotal = self.comput_metrix(cls_logits, batch['clslabels'])
        self.log('val_loss', loss)
        self.log('val_mask_acc', mask_acc)
        self.log('val_cls_acc', cls_acc)
        self.log('valid_acc', cls_acc)
    
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

        print("ncorrect = {}, ntotal = {}".format(ncorrect, ntotal))
        print(f"Validation Accuracy: {round(ncorrect / ntotal, 4)}")


    def predict_inputs(self, batch):
        #  Filter reduntant information(for example: 'sentence') that will be passed to model.forward()
        inputs = {
            'input_ids': batch['input_ids'].cuda(),
            'attention_mask': batch['attention_mask'].cuda(),
            'token_type_ids': batch['token_type_ids'].cuda(),
            "position_ids":batch['position_ids'].cuda(),
            "mlmlabels": batch['mlmlabels'].cuda(),
            "clslabels": batch['clslabels'].cuda(),
            "clslabels_mask": batch['clslabels_mask'].cuda(),
            "mlmlabels_mask": batch['mlmlabels_mask'].cuda(),
        }
        return inputs 

    def predict(self, batch):
        inputs = self.predict_inputs(batch)
        with torch.no_grad():
            loss, logits, cls_logits, hidden_states = self.model(**inputs)

        probs = torch.nn.functional.softmax(cls_logits, dim=-1)
        predicts = torch.argmax(probs, dim=-1)

        probs = probs.detach().cpu().numpy()
        predicts = predicts.detach().cpu().numpy()
        logits = cls_logits.detach().cpu().numpy()

        labels = None
        if batch["clslabels"] is not None:
            labels = batch["clslabels"].detach().cpu().numpy()

        # 转换为label_classes
        # label_idx = list(batch["label_idx"][0].numpy())

        probs_ = []
        predicts_ = []
        labels_ = []
        for label_idx,prob,predict,label in zip(batch["label_idx"], probs, predicts, labels):
            label_idx = list(label_idx.numpy())

            probs_.append([prob[i] for i in label_idx[:-1]])
            predicts_.append(label_idx.index(predict))
            labels_.append(label_idx.index(label))

        probs_ = np.array(probs_)

        sample_embeds = None
        if not batch["use_mask"][0]:
            # 获取样本的向量表示
            hidden_states = hidden_states.detach().cpu().numpy()
            input_ids = batch["input_ids"].detach().cpu().numpy()
            batch_size = input_ids.shape[0]
            sample_embeds = []
            for i in range(batch_size):
                # print("input_ids", input_ids[i])
                sep_token_indexes = np.where(input_ids[i] == self.sep_token)[0]
                sep_token_indexes = sep_token_indexes[-2:]
                if sep_token_indexes[0]+1 < sep_token_indexes[1]:
                    sample_hidden_states = hidden_states[i, sep_token_indexes[0]+1:sep_token_indexes[1], :]
                    sample_embed = np.mean(sample_hidden_states, axis=0)
                else:
                    sample_embed = np.zeros(hidden_states.shape[2])
                sample_embeds.append(sample_embed)
            sample_embeds = np.stack(sample_embeds, axis=0)

        return logits, probs_, predicts_, labels_, sample_embeds

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
            print("使用Adafactor!")
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

    def comput_metrix(self, logits, labels, mlmlabels_mask=None):
        logits = torch.nn.functional.softmax(logits, dim=-1)
        is_mlm = True if len(logits.shape) == 3 else False
        if is_mlm:
            batch_size, seq_len, hidden_size = logits.shape
            # logits = logits[:,:,yes_token]
            # ones = torch.ones_like(logits)
            # zero = torch.zeros_like(logits)
            # logits = torch.where(logits < 0.45, zero, ones)

            logits = logits[:,:,[self.no_token,self.yes_token]]
            logits = torch.argmax(logits, dim=-1)


            ones = torch.ones_like(labels)
            zero = torch.zeros_like(labels)
            labels = torch.where(labels < 3400, ones, zero)
            # print('labels',labels[0])
        else:
            logits = torch.argmax(logits, dim=-1)
            labels = labels

        y_pred = logits.view(size=(-1,))
        y_true = labels.view(size=(-1,))
        corr = torch.eq(y_pred, y_true).float()
        if is_mlm:
            corr = torch.multiply(mlmlabels_mask.view(-1,), corr)
            return torch.sum(corr.float())/torch.sum(mlmlabels_mask.float())
        else:
            return torch.sum(corr.float())/labels.size(0), torch.sum(corr.float()), labels.size(0)

