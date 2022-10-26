
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import argparse

import os
import uvicorn

from fastapi import FastAPI, Request
from fastapi import UploadFile, File
from starlette.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from enum import Enum
# from data_augmentation import get_eda,get_translation   
# from sim_gen_cpm.sim_data_cpm import get_cpm_sim
import numpy as np
import time
import requests
import json
from fastapi import FastAPI, File, UploadFile
from typing import List


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

app = FastAPI()
app.mount("/statics", StaticFiles(directory="statics"), name="statics")

# 创建一个templates（模板）对象，以后可以重用。

templates = Jinja2Templates(directory="statics/templates")
ip="192.168.190.2"
#ip = "192.168.190.63"



# -------------------------------------------主页---------------------------------------------------
@app.get('/',response_class=HTMLResponse)
async def index(request: Request):
    html_content = f"""<h1 align="center">欢迎来到IDEA认知计算组!</h1></br>" 
                <h1 align="center"><a href="http://{ip}:5201/docs#">在线调试算法服务api</a></h1>
                """
    return templates.TemplateResponse("index.html", {"request": request,"title":"GTS-teacher主页"})

# ------------------------------------------测试接口-------------------------------------------------
class People(BaseModel):
    name: str
    age: int



# ---------------------------------------实时log流---------------------------------------------------
# sends SSE events anytime logs are added to our log file.
from sse_starlette.sse import EventSourceResponse
from datetime import datetime
from sh import tail
from fastapi.middleware.cors import CORSMiddleware





from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import copy


@app.post('/api/upfiles/')
async def upload_files(upload_list:List[UploadFile]=File(...)):
    
    file_names = []
    file_types = []
    file_sizes = []

    for file_ in upload_list:
        
        contents = await file_.read()
        file_name = file_.filename
        file_type = file_.content_type

        
        #文件保存位置
        data_path = os.path.join(os.getcwd(),'data')
        if not os.path.exists(data_path):
            os.makedir(data_path)
        print('data_path', data_path)
        file_path = '{}/{}'.format(data_path, file_name)
        print('file_path', file_path)
        with open(file_path,'wb') as f:
            f.write(contents)

        file_names.append(file_name)
        file_types.append(file_type)


    return ({
            "file_names":file_names,
            "file_types":file_types
            })





# ------------------------------------------训练-------------------------------------------------

class TrainInput(BaseModel):
    train_data: str = "train_0.json"
    dev_data: str = "dev_0.json"
    test_data: str = "test_public.json"
    # data_dir: str = "/raid/liuyibo/gts_teacher_service/data"
    data_dir: str = "{}/data".format(os.getcwd())
    model_path: str = "{}/model_save".format(os.getcwd())
    train_batchsize: int = 1
    valid_batchsize: int = 4
    max_len: int = 512


# from train import start_train
@app.post('/api/train')
def train_data(inputs:TrainInput):
    # start_train()
    
    print('start')
    sh_command = "nohup python train.py \
                    --train_data={} \
                    --valid_data={} \
                    --test_data={} \
                    --data_dir={} \
                    --model_path={} \
                    --train_batchsize={} \
                    --valid_batchsize={} \
                    --max_len={} >log.txt &".format(inputs.train_data,
                                                    inputs.dev_data,
                                                    inputs.test_data,
                                                    inputs.data_dir,
                                                    inputs.model_path,
                                                    inputs.train_batchsize,
                                                    inputs.valid_batchsize,
                                                    inputs.max_len)

    print('sh_command', sh_command)
    os.system(sh_command)
    
    print('end')
    task_id =  int(os.getpid())
    
    # train.py创建task_id/data_dir/results/train_status.json
    # {"状态":"训练中"}
    # train.py训练完之后把改为{"状态":"已完成","acc":"100%"}
    return {"info":"已提交训练", "训练id":task_id}


@app.post('/api/task')
def task_status(task_id):
    # start_train()
    # 访问task_id/data_dir/results/train_status.json

    with open(os.path.join(os.getcwd(),'info.json'),'r') as f:
        res = json.loads(f.read())

    # train_status = open(task_id)
    return {"info":res['训练状态']}

 
# ------------------------------------------UnifiedMC-------------------------------------------------

from teacher_core.models.text_classification.bert_UnifiedMC import taskModel

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

    # print(args.task_name)

    return args

def load_tokenizer_and_model(checkpoint_path):
    
    args = load_args(checkpoint_path)
    save_path = os.path.split(checkpoint_path)[0]
    print("Load checkpoint from {}".format(checkpoint_path))

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(save_path)
    

    # 加载模型
    model = BertUnifiedMC.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer,load_from_tapt=False)
    # model=Bert.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
    model.eval()
    model = model.cuda()

    return tokenizer, model
    

# def predict(sentences,tuning_method, checkpoint_path,label_classes, batch_size):

class PredictInput(BaseModel):
    sentences: list = ["怎样的房子才算户型方正？","文登区这些公路及危桥将进入封闭施工，请注意绕行！"]
    checkpoint_path: int = ""


@app.post('/api/predict')
def predict(inputs:PredictInput):
# def predict(sentences, checkpoint_path, tokenizer, model):

    sentences = inputs.sentences
    checkpoint_path = inputs.checkpoint_path
    tokenizer, model = load_tokenizer_and_model(checkpoint_path)

    # checkpoint_path = '/raid/liuyibo/gts_teacher_service/save/output/epoch=00-valid_acc=0.6000-v2.ckpt'
    choice = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "股票", "农业", "电竞"]

    args = load_args(checkpoint_path)

    # 加载数据
    data_model = TaskDataModelUnifiedMC(args, tokenizer)

    samples = []
    question = "请问下面的文字描述属于那个类别？"
    #choice = list(data_model.label_classes.keys())
    for sentence in sentences:
        sample = {"id":0,"texta":sentence,"textb":"","question":question,"choice":choice,"label": 0}
        samples.append(sample)
    dataset = TaskDatasetUnifiedMC(data_path=None,args=args,used_mask=False, tokenizer=tokenizer, load_from_list=True, samples=samples)
    

    dataloader = DataLoader(dataset, shuffle=False, 
        collate_fn=data_model.collate_fn, \
        batch_size=args.train_batchsize)

    label_classes = data_model.label_classes
    print(label_classes)
    label_classes_reverse = {v:k for k,v in label_classes.items()}

    results = []

    # 进行预测
    for batch in dataloader:
        logits, probs, predicts, labels, _ = model.predict(batch)
    
        for idx, (predict,prob) in enumerate(zip(predicts,probs)):
            
            pred = {
                'sentence': batch['sentence'][idx],
                'label': predict,
                "label_name":choice[predict],
                "choice":choice,
                "probs":prob.tolist()
            }
            results.append(pred)
    return results







if __name__ == '__main__':
    # uvicorn.run("main:app", host='0.0.0.0', port=2333, reload=True, debug=True)
    # uvicorn.run(app, host='0.0.0.0', port=8080, debug = True)
    uvicorn.run(app, host='0.0.0.0', port=5201)
    # uvicorn.run(app, host='192.168.190.63', port=5201)
    # uvicorn.run(app, host='192.168.190.63', port=8080)
    # inputs = UnifiedMCInput()
    # res = zeroshot(inputs)
    # print(res)