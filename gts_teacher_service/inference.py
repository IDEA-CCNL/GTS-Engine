import os
import json
import torch
import argparse
from itertools import chain
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoModel, AutoTokenizer, AdamW, BertTokenizer


from teacher_core.utils.evaluation import evaluation
from teacher_core.dataloaders.text_classification.dataloader_UnifiedMC import TaskDatasetUnifiedMC
from teacher_core.models.text_classification.bert_UnifiedMC import BertUnifiedMC
from teacher_core.dataloaders.text_classification.dataloader_UnifiedMC import TaskDataModelUnifiedMC


import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, loggers
# from pytorch_lightning.callbacks.progress import tqdm
from tqdm.auto import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ObjDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value


def load_args(checkpoint_path):
    save_path = os.path.split(checkpoint_path)[0]

    args_dict = json.load(open(save_path+"/args.json"))

    args = ObjDict(args_dict)

    return args

def load_tokenizer_and_model(checkpoint_path):
    
    args = load_args(checkpoint_path)
    save_path = os.path.split(checkpoint_path)[0]
    print("Load checkpoint from {}".format(checkpoint_path))

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(save_path)
    
    # 加载模型
    model = BertUnifiedMC.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer,load_from_tapt=False)
    model.eval()
    model = model.cuda()

    return tokenizer, model

# def predict(sentences,tuning_method, checkpoint_path,label_classes, batch_size):
def predict(sentences, checkpoint_path, tokenizer, model):

    args = load_args(checkpoint_path)
    # 加载数据
    data_model = TaskDataModelUnifiedMC(args, tokenizer)

    samples = []
    question = "请问下面的文字描述属于那个类别？"
    choice = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "股票", "农业", "电竞"]

    for sentence in sentences:
        sample = {"id":0,"texta":sentence,"textb":"","question":question,"choice":choice,"label": 0}
        samples.append(sample)
    dataset = TaskDatasetUnifiedMC(data_path=None,args=args,used_mask=False, tokenizer=tokenizer, load_from_list=True, samples=samples)
    
    dataloader = DataLoader(dataset, shuffle=False, 
        collate_fn=data_model.collate_fn, \
        batch_size=args.train_batchsize)
    

    label_classes_reverse = {k:v for k,v in enumerate(choice)}

    results = []

    # 进行预测
    for batch in dataloader:

        logits, probs, predicts, labels, _ = model.predict(batch)
    
        for idx, (predict,prob) in enumerate(zip(predicts,probs)):
            
            pred = {
                'sentence': batch['sentence'][idx],
                'label': predict,
                "label_name":label_classes_reverse[predict],
                "probs":prob.tolist()
            }
            results.append(pred)
    
    return results



if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

    checkpoint_path = "/cognitive_comp/liuyibo/git_code/GTS-Engine/gts_teacher_service/save/epoch=00-valid_acc=0.6000.ckpt"

    sentences = [
    "为何农民工每天日夜加班却没有网红在家里直播几天的收入高？",
    "文登区这些公路及危桥将进入封闭施工，请注意绕行！"
    ]

    tokenizer, model = load_tokenizer_and_model(checkpoint_path=checkpoint_path)
    result = predict(sentences, checkpoint_path, tokenizer, model)
    print(result)



