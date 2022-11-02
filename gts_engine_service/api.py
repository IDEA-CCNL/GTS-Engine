
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
import psutil
import shutil
import subprocess
from fastapi import FastAPI, File, UploadFile
from typing import List
import api_utils
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
class CreateTaskInput(BaseModel):
    task_name: str # 任务名称
    task_type: str # 任务类型

@app.post('/api/create_task/')
def create_task(creat_task_input: CreateTaskInput):
    # task_type 可选:classification、similarity
    # 获得当前时间
    task_name = creat_task_input.task_name
    task_type = creat_task_input.task_type
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
            "task_name": task_name
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
class TaskStatusInput(BaseModel):
    task_id: str # 任务id

@app.post('/api/check_task_status')
def check_task_status(task_task_inut: TaskStatusInput):
    task_id = task_task_inut.task_id

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
class DeleteTaskInput(BaseModel):
    task_id: str # 任务id

@app.post('/api/delete_task/')
def delete_task(delete_task_input: DeleteTaskInput):
    task_id = delete_task_input.task_id

    task_dir = os.path.join(os.path.dirname(__file__), "tasks")
    if not api_utils.is_task_valid(task_dir, task_id):
        return {"ret_code": -100, "message": "task id不存在"}
    shutil.rmtree(os.path.join(task_dir, task_id))
    return {"ret_code": 200, "message": "Success"}


# ------------------------------------------模型训练-------------------------------------------------
class TrainInput(BaseModel):
    task_id: str # 任务id
    train_data: str # 训练集名称
    val_data: str # 验证集名称 
    test_data: str # 测试集名称
    label_data: str # 标签数据名称
    max_len: int = 512 # 文本最大长度
    max_num_epoch: int = 1 # 最大训练轮次
    min_num_epoch: int = 1 # 最小训练轮次
    seed: int = 42 # 随机种子
    gpuid: int 
    
        
@app.post('/api/train')
def start_train(train_input: TrainInput):
    task_dir = os.path.join(os.path.dirname(__file__), "tasks")
    task_id = train_input.task_id
    if not api_utils.is_task_valid(task_dir, task_id):
        return {"ret_code": -100, "message": "任务id不存在"}

    task_dir = os.path.join(task_dir, task_id)
    task_data_dir = os.path.join(task_dir, "data")
    
    task_info_path = os.path.join(task_dir, "task_info.json")
    if not os.path.exists(task_info_path):
        return {"ret_code": -102, "message": "任务信息文件不存在"}
    task_info = json.load(open(task_info_path))

    if not api_utils.is_data_format_valid(os.path.join(task_data_dir, train_input.train_data), "train"):
        return {"ret_code": -101, "message":"训练数据不存在或者数据格式不合法"}

    if not api_utils.is_data_format_valid(os.path.join(task_data_dir, train_input.val_data), "dev"):
        return {"ret_code": -101, "message":"验证数据不存在或者数据格式不合法"}

    if not api_utils.is_data_format_valid(os.path.join(task_data_dir, train_input.test_data), "test"):
        return {"ret_code": -101, "message":"测试数据不存在或者数据格式不合法"}

    if not api_utils.is_data_format_valid(os.path.join(task_data_dir, train_input.label_data), "label"):
        return {"ret_code": -101, "message":"标签数据不存在或者数据格式不合法"}

    # 创建日志目录和模型存储目录
    task_log_dir = os.path.join(task_dir, "logs")
    task_output_dir = os.path.join(task_dir, "outputs")
    if os.path.exists(task_log_dir):
        shutil.rmtree(task_log_dir)
    os.makedirs(task_log_dir)
    if os.path.exists(task_output_dir):
        shutil.rmtree(task_output_dir)
    os.makedirs(task_output_dir)

    train_batch_size = 1
    val_batch_size = 4
    val_check_interval = 0.25

    print('start training...')
    args = [
        "--task_dir=%s" % task_dir,
        "--train_data=%s" % train_input.train_data,
        "--valid_data=%s" % train_input.val_data,
        "--test_data=%s" % train_input.test_data,
        "--label_data=%s" % train_input.label_data,
        "--data_dir=%s" % task_data_dir,
        "--save_path=%s" % task_output_dir,
        "--train_batchsize=%d" % train_batch_size,
        "--valid_batchsize=%d" %  val_batch_size,
        "--max_len=%d" % train_input.max_len,
        "--max_epochs=%d" % train_input.max_num_epoch,
        "--min_epochs=%d" % train_input.min_num_epoch,
        "--seed=%d" % train_input.seed,
        "--val_check_interval=%f" % val_check_interval,
    ]

    proc_args = ["python", "train.py"] + args
    # proc = subprocess.Popen(proc_args)

    task_train_log = os.path.join(task_log_dir, "train.log")

    with open(task_train_log,"w") as writer:
        proc = subprocess.Popen(';'.join(['export CUDA_VISIBLE_DEVICES={}'.format(str(train_input.gpuid)), ' '.join(proc_args)]), shell=True, stdout=writer, stderr=writer)

    task_info["status"] = "On Training"
    task_info["status_code"] = 1
    task_info["train_pid"] = proc.pid
    task_info["train_data"] = train_input.train_data
    task_info["val_data"] = train_input.val_data
    task_info["test_data"] = train_input.test_data
    task_info["label_data"] = train_input.label_data
    with open(task_info_path, mode="w") as f:
            json.dump(task_info, f, indent=4)

    return {"ret_code": 200, "message": "训练调度成功"}
 
# ------------------------------------------停止模型训练-------------------------------------------------
class StopTrainInput(BaseModel):
    task_id: str # 任务id

@app.post('/api/stop_train')
def stop_train(stop_train_input: StopTrainInput):
    task_id = stop_train_input.task_id

    task_dir = os.path.join(os.path.dirname(__file__), "tasks")
    if not api_utils.is_task_valid(task_dir, task_id):
        return {"ret_code": -100, "message": "任务id不存在"}

    task_dir = os.path.join(task_dir, task_id)
    task_info_path = os.path.join(task_dir, "task_info.json")
    if not os.path.exists(task_info_path):
        return {"ret_code": -102, "message": "任务信息文件不存在"}
    task_info = json.load(open(task_info_path))

    if task_info["status"] != "On Training":
        return {"ret_code": -103, "message": "任务不在训练中"}

    proc = psutil.Process(task_info["train_pid"])
    proc.kill()
    print("train process %d is killed" % task_info["train_pid"])

    task_info["status"] = "Train Stopped"
    task_info["status_code"] = 3

    return {"ret_code": 200, "message": "终止训练成功"}

# ------------------------------------------开启模型预测-------------------------------------------------

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
from tqdm.auto import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class StartInferenceInput(BaseModel):
    task_id: str # 任务id


@app.post('/api/start_inference')
def start_inference(start_inference_input: StartInferenceInput):
    task_id = start_inference_input.task_id

    global inference_tokenizer
    global inference_model
    global inference_choice
    global inference_args

    task_info_path =  'tasks/{}/task_info.json'.format(task_id)

    if os.path.exists(task_info_path):
        task_info_dict = json.load(open(task_info_path,'r', encoding='utf-8'))
    else:
        return {"ret_code":-100, "message": "task_id not exits"}

    checkpoint_path =  task_info_dict['best_model_path']
    task_type = task_info_dict['task_type']
    label_data = task_info_dict["label_data"]

    label_path = os.path.join(os.path.dirname(__file__), "tasks", task_id,  "data", label_data)
    print("label_path",label_path)
    line = json.load(open(label_path, 'r', encoding='utf8'))
    inference_choice = line['labels']
    inference_args = api_utils.load_args(checkpoint_path)

    # tokenizer, model = load_tokenizer_and_model(checkpoint_path, inputs.tuning_method)
    inference_tokenizer, inference_model = api_utils.load_tokenizer_and_model(checkpoint_path)

    task_info_path = os.path.join(os.path.dirname(__file__), "tasks", task_id, "task_info.json")
    task_info = json.load(open(task_info_path))
    task_info["status"] = "Load Predict Model"
    with open(task_info_path, mode="w") as f:
        json.dump(task_info, f, indent=4)


    return {"ret_code":200, "message":"加载预测模型"}


# ------------------------------------------模型预测-------------------------------------------------
class PredictInput(BaseModel):
    sentences: list = ["怎样的房子才算户型方正？","文登区这些公路及危桥将进入封闭施工，请注意绕行！"]
    task_id: str

@app.post('/api/predict')
def predict(inputs:PredictInput):

    sentences = inputs.sentences
    task_info_path =  'tasks/{}/task_info.json'.format(inputs.task_id)
    if os.path.exists(task_info_path):
        task_info_dict = json.load(open(task_info_path,'r', encoding='utf-8'))
    else:
        return {"ret_code":-100, "message": "task_id not exits"}
    
    # checkpoint_path =  task_info_dict['best_model_path']
    # label_data = task_info_dict["label_data"]
    # label_path = os.path.join(os.path.dirname(__file__), "tasks", inputs.task_id,  "data", label_data)
    # print("label_path",label_path)
    # line = json.load(open(label_path, 'r', encoding='utf8'))


    # choice = line['labels']
    # args = api_utils.load_args(checkpoint_path)
    
    # 加载数据
    data_model = TaskDataModelUnifiedMC(inference_args, inference_tokenizer)

    samples = []
    question = "请问下面的文字描述属于那个类别？"

    for sentence in sentences:
        sample = {"id":0, "content":sentence, "textb":"", "question":question, "choice":inference_choice, "label":inference_choice[0]}
        samples.append(sample)
    dataset = TaskDatasetUnifiedMC(data_path=None, args=inference_args, used_mask=False, tokenizer=inference_tokenizer, load_from_list=True, samples=samples, choice=inference_choice)
    
    dataloader = DataLoader(dataset, shuffle=False, 
        collate_fn=data_model.collate_fn, \
        batch_size=inference_args.train_batchsize)

    label_classes = data_model.label_classes
    print(label_classes)
    label_classes_reverse = {v:k for k,v in label_classes.items()}

    pred_labels = []
    pred_probs = []

    for batch in dataloader:
        logits, probs, predicts, labels, _ = inference_model.predict(batch)
    
        for idx, (predict,prob) in enumerate(zip(predicts,probs)):    
            pred_labels.append(inference_choice[predict])
            pred_probs.append(prob.tolist())

    return {'ret_code':200, 'predictions':pred_labels, 'probabilities':pred_probs}


# ------------------------------------------关闭模型预测-------------------------------------------------
class EndInferenceInput(BaseModel):
    task_id: str  #任务id

@app.post('/api/end_inference')
def end_inference(end_inference_input: EndInferenceInput):
    task_id = end_inference_input.task_id

    global inference_tokenizer
    global inference_model
    del inference_model
    del inference_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    task_info_path = os.path.join(os.path.dirname(__file__), "tasks", task_id, "task_info.json")
    task_info = json.load(open(task_info_path))
    task_info["status"] = "Release Predict Model"
    with open(task_info_path, mode="w") as f:
        json.dump(task_info, f, indent=4)

    return {'ret_code':200, "message":"释放预测模型"}



if __name__ == '__main__':
    # uvicorn.run(app, host='0.0.0.0', port=8080, debug = True)
    # uvicorn.run(app, host='0.0.0.0', port=5201)
    uvicorn.run(app, host='0.0.0.0', port=5201)
    # uvicorn.run(app, host='192.168.190.63', port=5201)
