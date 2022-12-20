import os
import sys 
import json
import torch
import shutil
import pickle
import argparse
import traceback
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from transformers import AutoModel, AutoTokenizer, BertTokenizer, MegatronBertForMaskedLM

from pytorch_lightning import Trainer, seed_everything, loggers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 如果没有安装gts_engine，请把GTS-Engine/gts-engine加入到系统环境变量
sys.path.append(os.path.dirname(__file__))

from pipelines import *
from qiankunding.utils.tokenization import get_train_tokenizer
from qiankunding.utils import knn_utils

from qiankunding.dataloaders.nli.dataloader_UnifiedMC import TaskDataModelUnifiedMCForNLI
from qiankunding.models.nli.bert_UnifiedMC import BertUnifiedMCForNLI

from gts_common.registry import PIPELINE_REGISTRY
from gts_common.arguments import GtsEngineArgs
# 设置gpu相关的全局变量
import qiankunding.utils.globalvar as globalvar
globalvar._init()
from qiankunding.utils.detect_gpu_memory import detect_gpu_memory, decide_gpu
from gts_common.logs_utils import Logger

logger = Logger().get_log()

gpu_memory, gpu_cur_used_memory = detect_gpu_memory()
globalvar.set_value("gpu_type", decide_gpu(gpu_memory))
globalvar.set_value('gpu_max_used_memory', gpu_cur_used_memory)

def train(args):
    # Set global random seed
    seed_everything(args.seed)
    
    # Set path to save checkpoint and outputs
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.output_dir = save_path

    train_pipeline_module = "pipelines." + args.engine_type + "_" + args.task_type
    train_pipeline = PIPELINE_REGISTRY.get(name="train_pipeline", suffix=train_pipeline_module)
    train_pipeline(args)

def main():
    total_parser = argparse.ArgumentParser()

    total_parser.add_argument("--engine_type", required=True, choices=["qiankunding", "bagualu"],
                            type=str, help="engine type")
    total_parser.add_argument("--train_mode", required=True, choices=["fast", "standard", "advanced"],
                            type=str, help="training mode")
    total_parser.add_argument("--task_dir", required=True, 
                            type=str, help="specific task directory")
    total_parser.add_argument("--task_type", required=True, choices=["classification", "similarity", "nli", "ie"],
                            type=str, help="task type for training")
    total_parser.add_argument('--num_workers', default=8,
                            type=int, help="number of workers for data preprocessing.")
    total_parser.add_argument('--train_batchsize', default=1,
                            type=int, help="batch size of train dataset")   
    total_parser.add_argument('--valid_batchsize', default=4,
                            type=int, help="batch size of validation dataset")   
    total_parser.add_argument('--test_batchsize', default=4,
                            type=int, help="batch size of test dataset")  
    total_parser.add_argument('--max_len', default=512,
                            type=int, help="max length of input text")   

    total_parser.add_argument('--pretrained_model_dir', required=True,
                            type=str, help="path to the directory which contains all the pretrained models downloaded from huggingface")
    total_parser.add_argument('--data_dir', required=True,
                            type=str, help="data directory of specific task data")
    total_parser.add_argument('--train_data', required=True,
                            type=str, help="filename of train dataset")   
    total_parser.add_argument('--valid_data', required=True,
                            type=str, help="filename of validation dataset")     
    total_parser.add_argument('--test_data', default='test.json',
                            type=str, help="filename of test dataset")      
    total_parser.add_argument('--label_data', default='labels.json',
                            type=str, help="filename of label data")
    total_parser.add_argument('--unlabeled_data', default='test.json', 
                            type=str, help="filename of unlabeled data")   #unlabeled.json
    total_parser.add_argument('--save_path', default='output',
                            type=str, help="save path for trained model and other logs")
    total_parser.add_argument('--threshold', default=0.8,
                            type=float, help="pseudo threshold")

    # * Args for general setting
    total_parser.add_argument('--seed', default=1234,
                            type=int, help="random seed for training")
    total_parser.add_argument('--lr', default=2e-5,
                            type=float, help="learning rate")
    
    # * Args for Trainer
    total_parser.add_argument('--max_epochs', default=None,
                              type=int, help="upper limit of training epochs")
    total_parser.add_argument('--min_epochs', default=None,
                              type=int, help="lower limit of training epochs")
    total_parser.add_argument('--val_check_interval', default=0.5,
                              type=float, help="perform a validation loop every after every `N` training epochs")

    print("total_parser:",total_parser)
    args = total_parser.parse_args(namespace=GtsEngineArgs())

    logger.info("pretrained_model_dir {}".format(args.pretrained_model_dir))
    args.gpus = 1
    logger.info("args {}".format(args))
    torch.set_num_threads(8)
    

    task_info_path = os.path.join(args.task_dir, "task_info.json")
    if os.path.exists(task_info_path):
        task_info = json.load(open(task_info_path))
    else:
        task_info = {}

    task_info["status"] = "On Training"
    task_info["status_code"] = 1
    task_info["train_pid"] = os.getpid() 
    task_info["train_data"] = args.train_data
    task_info["val_data"] = args.valid_data
    task_info["test_data"] = args.test_data
    task_info["label_data"] = args.label_data
    with open(task_info_path, mode="w") as f:
            json.dump(task_info, f, indent=4)

    try:
        train(args)
        task_info = json.load(open(task_info_path))
        task_info["status"] = "Train Success"
        task_info["status_code"] = 2
        task_info["save_path"] = args.save_path
        with open(task_info_path, mode="w") as f:
            json.dump(task_info, f, indent=4)
    except:
        traceback.print_exc()
        task_info["status"] = "Train Failed"
        task_info["status_code"] = 3
        with open(task_info_path, mode="w") as f:
            json.dump(task_info, f, indent=4)

if __name__ == '__main__':    
    main()
    