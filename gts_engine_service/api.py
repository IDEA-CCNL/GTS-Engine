
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, Form
from starlette.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from enum import Enum
import time
import json
import datetime
import shutil
from fastapi import FastAPI, File, UploadFile
from typing import List
import api_utils
from teacher_config import tuning_methods_config
import gc
app = FastAPI()

# -------------------------------------------主页---------------------------------------------------
@app.get('/',response_class=HTMLResponse)
async def index(request: Request):
    
    html_content = f""" 
                <h1 align="center"></h1></br>
                <h1 align="center"></h1></br>
                <h1 align="center">GTS-Engine在线算法API</h1>
                """

    return html_content

# ---------------------------------------创建任务---------------------------------------------------
@app.post('/api/create_task/')
def create_task(task_name: str, task_type: str, task_method: str):
    # task_method 可选:UnifiedMC、promtbert
    # task_type 可选:classification、similarity
    # 获得当前时间
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if task_name is None:
        task_name = task_type
    task_id = task_name + "_" + timestamp_str
    task_dir = os.path.join(os.path.dirname(__file__), "tasks")
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    if api_utils.is_task_valid(task_dir, task_id):
        return {"ret_code": -100, "message": "task已经存在", "task_id": task_id}
    else:
        task_info = {
            "task_id": task_id,
            "status": "Initialized",
            "status_code": 0,
            "task_type": task_type,
            "task_name": task_name,
            "task_method": task_method
        }
        specific_task_dir = os.path.join(task_dir, task_id)
        if not os.path.exists(specific_task_dir):
            os.makedirs(specific_task_dir)
        with open(os.path.join(specific_task_dir, "task_info.json"), mode="w") as f:
            json.dump(task_info, f, indent=4)
        return {"ret_code": 200, "message": "task成功创建", "task_id": task_id}
    

# ---------------------------------------创建任务---------------------------------------------------
@app.post('/api/list_task/')
def list_task():
    task_dir = os.path.join(os.path.dirname(__file__), "tasks")
    tasks = api_utils.list_task(task_dir)
    return {"ret_code": 200, "message": "Success", "tasks": tasks}

# ------------------------------------------查看任务状态-------------------------------------------------
@app.post('/api/check_task_status')
def check_task_status(task_id: str):
    task_dir = os.path.join(os.path.dirname(__file__), "tasks")
    if not api_utils.is_task_valid(task_dir, task_id):
        return {"ret_code": -100, "message": "任务不存在"}
    specific_task_dir = os.path.join(task_dir, task_id)
    task_info_path = os.path.join(specific_task_dir, "task_info.json")
    if not os.path.exists(task_info_path):
        return {"ret_code": -200, "message": "任务信息文件不存在"}
    task_info = json.load(open(task_info_path))
    status = task_info["status"]
    status_code = task_info["status_code"]
    return {"ret_code": status_code, "message": status}

# ---------------------------------------文件上传---------------------------------------------------
@app.post('/api/upfiles/')
async def upload_files(files:List[UploadFile]=File(...), task_id: str = Form()):
    task_dir = os.path.join(os.path.dirname(__file__), "tasks")
    if not api_utils.is_task_valid(task_dir, task_id):
        return {"ret_code": -100, "message": "task id不存在"}

    task_dir = os.path.join(task_dir, task_id)
    data_dir = os.path.join(task_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print('file will save to', data_dir)

    for file_ in files:
        contents = await file_.read()
        file_name = file_.filename
        
        file_path = os.path.join(data_dir, file_name)
        print('file_path', file_path)
        with open(file_path, 'wb') as f:
            f.write(contents)

    return {"ret_code": 200, "message": "上传成功"}

# ---------------------------------------创建任务---------------------------------------------------
@app.post('/api/delete_task/')
def delete_task(task_id: str):
    task_dir = os.path.join(os.path.dirname(__file__), "tasks")
    if not api_utils.is_task_valid(task_dir, task_id):
        return {"ret_code": -100, "message": "task id不存在"}
    shutil.rmtree(os.path.join(task_dir, task_id))
    return {"ret_code": 200, "message": "Success"}

# ------------------------------------------模型训练-------------------------------------------------

class TrainInput(BaseModel):
    task_id: str = ""
    train_data: str = "train.json"
    valid_data: str = "dev.json"
    test_data: str = "test.json"
    label_data: str = "labels.json"
    
    # data_dir: str = "{}/files/data".format(os.getcwd())
    # save_path: str = "{}/files".format(os.getcwd())
    
    train_batchsize: int = 1
    valid_batchsize: int = 4
    max_len: int = 512
    # task_type: str = "classification"
    # tuning_method: str = "UnifiedMC"
    max_epochs: int = 10
    min_epochs: int = 1 
    num_threads: int = 8 
    seed: int = 123 
    val_check_interval: float = 0.25
    gpuid: int = 0
    pretrained_model_dir: str = "/raid/liuyibo/GTS-Engine/"


def check_train_data(data_path, data_type):
    is_correct = True
    try:
        with open(data_path, 'r', encoding='utf8') as f:
            data = f.readlines()
            for line in data:
                try:
                    line = json.loads(line)
                except:
                    is_correct = False

                if data_type=='train' or data_type=='dev':
                    if "text" not in line or  "label" not in line:
                        is_correct = False
                if data_type=='test':
                    if "text" not in line:
                        is_correct = False
                if data_type=='label':
                    if "labels" not in line:
                        is_correct = False 
    except:
        is_correct = False
    return is_correct
        

@app.post('/api/train')
def train_data(inputs:TrainInput):
    if not inputs.task_id:
        return {"ret_code": -100, "message": "please enter task_id"}

    train_data_path = os.path.join(os.path.dirname(__file__), "tasks", inputs.task_id, "data", inputs.train_data)
    if not check_train_data(train_data_path, "train"):
        return {"message":"请检查训练数据的地址是否存在或者数据格式是否符合要求"}

    val_data_path = os.path.join(os.path.dirname(__file__), "tasks", inputs.task_id, "data", inputs.valid_data)
    if not check_train_data(val_data_path, "dev"):
        return {"message":"请检查验证数据的地址是否存在或者数据格式是否符合要求"}

    if inputs.test_data:
        test_data_path = os.path.join(os.path.dirname(__file__), "tasks", inputs.task_id, "data", inputs.test_data)
        if not check_train_data(test_data_path, "test"):
            return {"message":"请检查测试数据的地址是否存在或者数据格式是否符合要求"}

    label_data_path = os.path.join(os.path.dirname(__file__), "tasks", inputs.task_id, "data", inputs.label_data)
    if not check_train_data(label_data_path, "label"):
        return {"message":"请检查标签数据的地址是否存在或者数据格式是否符合要求"}



    task_info_path =  'tasks/{}/task_info.json'.format(inputs.task_id)
    if os.path.exists(task_info_path):
        task_info_dict = json.load(open(task_info_path,'r', encoding='utf-8'))
    else:
        return {"ret_code":-100, "message": "task_id not exits"}

    tuning_method = task_info_dict['task_method']
    task_type = task_info_dict['task_type']
    
    if tuning_method not in ["UnifiedMC"]:
        return {"error":"tuning_method must be UnifiedMC"}
    if task_type not in ["classification"]:
        return {"error":"task_type must be classification"}

    print('start')

    data_dir = "tasks/{}/data".format(inputs.task_id)
    save_path = "tasks/{}/models".format(inputs.task_id)

    sh_command = "CUDA_VISIBLE_DEVICES={} nohup python train.py \
                    --task_id={} \
                    --train_data={} \
                    --valid_data={} \
                    --test_data={} \
                    --label_data={} \
                    --data_dir={} \
                    --save_path={} \
                    --train_batchsize={} \
                    --valid_batchsize={} \
                    --max_len={} \
                    --tuning_method={} \
                    --max_epochs={} \
                    --min_epochs={} \
                    --num_threads={} \
                    --seed={} \
                    --val_check_interval={} \
                    --pretrained_model_dir={} > tasks/{}/train_{}.log &".format(
                                                    inputs.gpuid,
                                                    inputs.task_id,
                                                    inputs.train_data,
                                                    inputs.valid_data,
                                                    inputs.test_data,
                                                    inputs.label_data,
                                                    data_dir,
                                                    save_path,
                                                    inputs.train_batchsize,
                                                    inputs.valid_batchsize,
                                                    inputs.max_len,
                                                    tuning_method,
                                                    inputs.max_epochs,
                                                    inputs.min_epochs,
                                                    inputs.num_threads,
                                                    inputs.seed,
                                                    inputs.val_check_interval,
                                                    inputs.pretrained_model_dir,
                                                    inputs.task_id,
                                                    inputs.task_id)

    print('sh_command', sh_command)
    os.system(sh_command)
    
    task_info_path =  'tasks/{}/task_info.json'.format(inputs.task_id)

    if os.path.exists(task_info_path):
        task_info_dict = json.load(open(task_info_path,'r', encoding='utf-8'))

        task_info_dict['label_path'] = os.path.join(data_dir, inputs.label_data)
        task_info_dict['train_path'] = os.path.join(data_dir, inputs.valid_data)
        task_info_dict['valid_path'] = os.path.join(data_dir, inputs.label_data)
        task_info_dict['test_path'] = os.path.join(data_dir, inputs.test_data)

        with open(task_info_path, mode="w") as f:
                json.dump(task_info_dict, f, indent=4)

    # task_id =  int(os.getpid())
    # train.py创建task_id/data_dir/results/train_status.json
    # {"状态":"训练中"}
    # train.py训练完之后把改为{"状态":"已完成","acc":"100%"}

    return {"ret_code": 200, "message": "Success"}

 
# ------------------------------------------模型预测-------------------------------------------------

from itertools import chain
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoModel, AutoTokenizer, AdamW, BertTokenizer

from teacher_core.utils.evaluation import evaluation
from teacher_core.dataloaders.text_classification.dataloader_UnifiedMC import TaskDatasetUnifiedMC
from teacher_core.models.text_classification.bert_UnifiedMC import taskModel, BertUnifiedMC
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

    return args


def load_tokenizer_and_model(checkpoint_path, tuning_method):
    global inference_tokenizer
    global inference_model
    args = load_args(checkpoint_path)
    save_path = os.path.split(checkpoint_path)[0]
    print("Load checkpoint from {}".format(checkpoint_path))

    # 加载tokenizer
    inference_tokenizer = BertTokenizer.from_pretrained(save_path)

    # 加载模型
    inference_model = tuning_methods_config[tuning_method]["TuningModel"].load_from_checkpoint(checkpoint_path, tokenizer=inference_tokenizer,load_from_tapt=False)
    # model=Bert.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
    inference_model.eval()
    inference_model = inference_model.cuda()

    return inference_tokenizer, inference_model
    

class PredictInput(BaseModel):
    sentences: list = ["怎样的房子才算户型方正？","文登区这些公路及危桥将进入封闭施工，请注意绕行！"]
    # label_set: str = "labels.json"
    # task_type: str = "text_classification"
    # tuning_method: str = "UnifiedMC"
    task_id: str = ""


@app.post('/api/start_inference')
# def start_inference(inputs:PredictInput):
def start_inference(task_id:str):

    task_info_path =  'tasks/{}/task_info.json'.format(task_id)

    if os.path.exists(task_info_path):
        task_info_dict = json.load(open(task_info_path,'r', encoding='utf-8'))
    else:
        return {"ret_code":-100, "message": "task_id not exits"}

    checkpoint_path =  task_info_dict['best_model_path']
    task_type = task_info_dict['task_type']
    tuning_method = task_info_dict['task_method']


    # if inputs.tuning_method not in ["UnifiedMC"]:
    #     return {"error": "tuning_method must be UnifiedMC"}
    # if inputs.task_type not in ["classification"]:
    #     return {"error": "task_type must be classification"}


    # tokenizer, model = load_tokenizer_and_model(checkpoint_path, inputs.tuning_method)
    inference_tokenizer, inference_model = load_tokenizer_and_model(checkpoint_path, tuning_method)

    return {"ret_code":200}



@app.post('/api/predict')
def predict(inputs:PredictInput):

    sentences = inputs.sentences
    task_info_path =  'tasks/{}/task_info.json'.format(inputs.task_id)
    if os.path.exists(task_info_path):
        task_info_dict = json.load(open(task_info_path,'r', encoding='utf-8'))
    else:
        return {"ret_code":-100, "message": "task_id not exits"}
    checkpoint_path =  task_info_dict['best_model_path']
    tuning_method = task_info_dict['task_method']
    labels_path = task_info_dict["label_path"]

    # tokenizer, model = load_tokenizer_and_model(checkpoint_path, inputs.tuning_method)
    #labels_path = "tasks/{}/data/{}".format(inputs.task_id, inputs.label_set)

    print("labels_path",labels_path)
    line = json.load(open(labels_path, 'r', encoding='utf8'))
    choice = line['labels']
    args = load_args(checkpoint_path)
    

    # 加载数据
    data_model = tuning_methods_config[tuning_method]["DataModel"](args, inference_tokenizer)

    if tuning_method == "UnifiedMC":
        samples = []
        question = "请问下面的文字描述属于那个类别？"

        for sentence in sentences:
            sample = {"id":0,"text":sentence,"textb":"","question":question,"choice":choice,"label": choice[0]}
            samples.append(sample)
        dataset = TaskDatasetUnifiedMC(data_path=None,args=args,used_mask=False, tokenizer=inference_tokenizer, load_from_list=True, samples=samples, choice=choice)
    

    dataloader = DataLoader(dataset, shuffle=False, 
        collate_fn=data_model.collate_fn, \
        batch_size=args.train_batchsize)

    label_classes = data_model.label_classes
    print(label_classes)
    label_classes_reverse = {v:k for k,v in label_classes.items()}

    results = []

    # 进行预测
    for batch in dataloader:
        logits, probs, predicts, labels, _ = inference_model.predict(batch)
    
        for idx, (predict,prob) in enumerate(zip(predicts,probs)):
            
            pred = {
                'sentence': batch['sentence'][idx],
                'label': predict,
                "label_name":choice[predict],
                # "choice":choice,
                "probs":prob.tolist()
            }
            results.append(pred)
    return {'ret_code':200, "results": results}



@app.post('/api/end_inference')
def end_inference():
    global inference_tokenizer
    global inference_model
    del inference_model
    del inference_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return {'ret_code':200}



if __name__ == '__main__':
    # uvicorn.run(app, host='0.0.0.0', port=8080, debug = True)
    # uvicorn.run(app, host='0.0.0.0', port=5201)
    uvicorn.run(app, host='0.0.0.0', port=5201)
    # uvicorn.run(app, host='192.168.190.63', port=5201)