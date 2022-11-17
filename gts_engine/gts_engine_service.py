
import torch
import argparse
import os
import sys
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, Form
from starlette.responses import HTMLResponse
from pydantic import BaseModel
import json
import datetime
import psutil
import shutil
import subprocess
from fastapi import FastAPI, File, UploadFile
from typing import List

# 如果没有安装gts_engine，请把GTS-Engine/gts-engine加入到系统环境变量
sys.path.append(os.path.dirname(__file__))

from gts_common import service_utils
from gts_engine_inference import preprare_inference, inference_samples
import gc

app = FastAPI()

## 全局参数
TASK_DIR = None # 任务存放的目录
PRETRAINED_DIR = None # 预训练模型存放的目录

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
    task_name: str = "" # 任务名称
    task_type: str = "" # 任务类型

@app.post('/api/create_task/')
def create_task(create_task_input: CreateTaskInput):
    task_name = create_task_input.task_name
    task_type = create_task_input.task_type
    if not task_name:
        task_name = task_type + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    task_id = task_name # 任务名称等于任务id
    if not os.path.exists(TASK_DIR):
        os.makedirs(TASK_DIR)
    if service_utils.is_task_valid(TASK_DIR, task_id):
        return {"ret_code": -100, "message": "task已经存在", "task_id": task_id}
    else:
        task_info = {
            "task_id": task_id,
            "status": "Initialized",
            "status_code": 0,
            "task_type": task_type,
            "task_name": task_name
        }
        specific_task_dir = os.path.join(TASK_DIR, task_id)
        if not os.path.exists(specific_task_dir):
            os.makedirs(specific_task_dir)
        with open(os.path.join(specific_task_dir, "task_info.json"), mode="w") as f:
            json.dump(task_info, f, indent=4)
        return {"ret_code": 200, "message": "task成功创建", "task_id": task_id}
    

# ---------------------------------------查看任务列表---------------------------------------------------
@app.post('/api/list_task/')
def list_task():
    tasks = service_utils.list_task(TASK_DIR)
    return {"ret_code": 200, "message": "Success", "tasks": tasks}


# ------------------------------------------查看任务状态-------------------------------------------------
class CheckTaskInput(BaseModel):
    task_id: str = "" # 任务id

@app.post('/api/check_task_status')
def check_task_status(check_task_input: CheckTaskInput):
    task_id = check_task_input.task_id
    if not service_utils.is_task_valid(TASK_DIR, task_id):
        return {"ret_code": -100, "message": "任务不存在"}
    specific_task_dir = os.path.join(TASK_DIR, task_id)
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
    if not service_utils.is_task_valid(TASK_DIR, task_id):
        return {"ret_code": -100, "message": "task id不存在"}

    specific_task_dir = os.path.join(TASK_DIR, task_id)
    data_dir = os.path.join(specific_task_dir, "data")
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


# ---------------------------------------删除任务---------------------------------------------------
class DeleteTaskInput(BaseModel):
    task_id: str = "" # 任务id

@app.post('/api/delete_task/')
def delete_task(delete_task_input: DeleteTaskInput):
    task_id = delete_task_input.task_id
    if not service_utils.is_task_valid(TASK_DIR, task_id):
        return {"ret_code": -100, "message": "task id不存在"}
    shutil.rmtree(os.path.join(TASK_DIR, task_id))
    return {"ret_code": 200, "message": "Success"}


# ------------------------------------------模型训练-------------------------------------------------
class TrainInput(BaseModel):
    task_id: str = ""# 任务id
    train_data: str = "" # 训练集名称
    val_data: str = "" # 验证集名称 
    test_data: str = "" # 测试集名称
    label_data: str = "" # 标签数据名称
    max_len: int = 512 # 文本最大长度
    max_num_epoch: int = 1 # 最大训练轮次
    min_num_epoch: int = 1 # 最小训练轮次
    seed: int = 42 # 随机种子
    gpuid: int = 0
    
        
@app.post('/api/train')
def start_train(train_input: TrainInput):
    task_id = train_input.task_id
    if not service_utils.is_task_valid(TASK_DIR, task_id):
        return {"ret_code": -100, "message": "任务id不存在"}

    specific_task_dir = os.path.join(TASK_DIR, task_id)
    task_data_dir = os.path.join(specific_task_dir, "data")
    
    task_info_path = os.path.join(specific_task_dir, "task_info.json")
    if not os.path.exists(task_info_path):
        return {"ret_code": -102, "message": "任务信息文件不存在"}
    task_info = json.load(open(task_info_path))

    if not service_utils.is_data_format_valid(os.path.join(task_data_dir, train_input.train_data), "train"):
        return {"ret_code": -101, "message":"训练数据不存在或者数据格式不合法"}

    if not service_utils.is_data_format_valid(os.path.join(task_data_dir, train_input.val_data), "dev"):
        return {"ret_code": -101, "message":"验证数据不存在或者数据格式不合法"}

    if not service_utils.is_data_format_valid(os.path.join(task_data_dir, train_input.test_data), "test"):
        return {"ret_code": -101, "message":"测试数据不存在或者数据格式不合法"}

    if not service_utils.is_data_format_valid(os.path.join(task_data_dir, train_input.label_data), "label"):
        return {"ret_code": -101, "message":"标签数据不存在或者数据格式不合法"}

    # 创建日志目录和模型存储目录
    task_log_dir = os.path.join(specific_task_dir, "logs")
    task_output_dir = os.path.join(specific_task_dir, "outputs")
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
        "--task_dir=%s" % specific_task_dir,
        "--task_type=%s" % task_info["task_type"],
        "--train_data=%s" % train_input.train_data,
        "--valid_data=%s" % train_input.val_data,
        "--test_data=%s" % train_input.test_data,
        "--label_data=%s" % train_input.label_data,
        "--pretrained_model_dir=%s" % PRETRAINED_DIR,
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

    train_script = os.path.join(os.path.dirname(__file__), "gts_engine_train.py")
    proc_args = ["python", train_script] + args
    # proc = subprocess.Popen(proc_args)

    task_train_log = os.path.join(task_log_dir, "train.log")

    with open(task_train_log,"w") as writer:
        proc = subprocess.Popen('; '.join(['export CUDA_VISIBLE_DEVICES={}'.format(str(train_input.gpuid)), ' '.join(proc_args)]), shell=True, stdout=writer, stderr=writer)
        

    return {"ret_code": 200, "message": "训练调度成功"}
 
# ------------------------------------------停止模型训练-------------------------------------------------
class StopTrainInput(BaseModel):
    task_id: str = "" # 任务id

@app.post('/api/stop_train')
def stop_train(stop_train_input: StopTrainInput):
    task_id = stop_train_input.task_id

    if not service_utils.is_task_valid(TASK_DIR, task_id):
        return {"ret_code": -100, "message": "任务id不存在"}

    specific_task_dir = os.path.join(TASK_DIR, task_id)
    task_info_path = os.path.join(specific_task_dir, "task_info.json")
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
    with open(task_info_path, mode="w") as f:
        json.dump(task_info, f, indent=4)

    return {"ret_code": 200, "message": "终止训练成功"}

# ------------------------------------------开启模型预测-------------------------------------------------
class StartInferenceInput(BaseModel):
    task_id: str = "" # 任务id


@app.post('/api/start_inference')
def start_inference(start_inference_input: StartInferenceInput):
    task_id = start_inference_input.task_id

    global inference_suite

    specific_task_dir = os.path.join(TASK_DIR, task_id)
    task_info_path = os.path.join(specific_task_dir, "task_info.json")
    if os.path.exists(task_info_path):
        task_info = json.load(open(task_info_path,'r', encoding='utf-8'))
    else:
        return {"ret_code":-100, "message": "task_id not exits"}

    save_path = task_info['save_path']
    task_type = task_info["task_type"]

    inference_suite = preprare_inference(task_type, save_path)

    task_info["status"] = "On Inference"
    task_info["statue_code"] = 3
    with open(task_info_path, mode="w") as f:
        json.dump(task_info, f, indent=4)

    return {"ret_code":200, "message":"加载预测模型"}


# ------------------------------------------模型预测-------------------------------------------------
class PredictInput(BaseModel):
    sentences: list = [{"content":"怎样的房子才算户型方正？"}, {"content":"文登区这些公路及危桥将进入 封闭施工，请注意绕行！"}]
    task_id: str = ""

@app.post('/api/predict')
def predict(inputs: PredictInput):
    sentences = inputs.sentences

    specific_task_dir = os.path.join(TASK_DIR, inputs.task_id)
    task_info_path = os.path.join(specific_task_dir, "task_info.json")
    if os.path.exists(task_info_path):
        task_info = json.load(open(task_info_path,'r', encoding='utf-8'))
    else:
        return {"ret_code":-100, "message": "task_id not exits"}
    
    task_type = task_info["task_type"]
    
    for sentence in sentences:
        if not isinstance(sentence, dict):
            return {"ret_code": -101, "message": "每条样本必须是字典形式"}
        else:
            if "content" not in sentence:
                return {"ret_code": -101, "message": "每条样本里必须包含content字段"}

    result = inference_samples(task_type, sentences, inference_suite)

    return {'ret_code':200, "result": result, "message": "预测成功"}


# ------------------------------------------关闭模型预测-------------------------------------------------
class EndInferenceInput(BaseModel):
    task_id: str = ""  #任务id

@app.post('/api/end_inference')
def end_inference(end_inference_input: EndInferenceInput):
    task_id = end_inference_input.task_id

    global inference_suite
    del inference_suite
    gc.collect()
    torch.cuda.empty_cache()

    specific_task_dir = os.path.join(TASK_DIR, task_id)
    task_info_path = os.path.join(specific_task_dir, "task_info.json")

    task_info = json.load(open(task_info_path))
    task_info["status"] = "Train Success"
    task_info["status_code"] = 2
    with open(task_info_path, mode="w") as f:
        json.dump(task_info, f, indent=4)

    return {'ret_code':200, "message":"释放预测模型"}

def main():
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--port', default=5201, type=int)
    arg_parser.add_argument('--task_dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks"), type=str)
    arg_parser.add_argument('--pretrained_dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained"), type=str)
    args = arg_parser.parse_args()

    global TASK_DIR
    TASK_DIR = args.task_dir
    global PRETRAINED_DIR
    PRETRAINED_DIR = args.pretrained_dir

    uvicorn.run(app, host='0.0.0.0', port=args.port)

if __name__ == '__main__':
    main()
