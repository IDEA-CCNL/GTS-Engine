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

from qiankunding_core.utils.evaluation import evaluation,Evaluator,SentencePairEvaluator
from qiankunding_core.utils.tokenization import get_train_tokenizer

from qiankunding_core.utils import knn_utils

from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 如果没有安装gts_engine，请把GTS-Engine/gts-engine加入到系统环境变量
sys.path.append(os.path.dirname(__file__))

from qiankunding_core.utils.evaluation import evaluation
from qiankunding_core.utils.tokenization import get_train_tokenizer
from qiankunding_core.utils import knn_utils
from qiankunding_core.dataloaders.text_classification.dataloader_UnifiedMC import TaskDataModelUnifiedMC
from qiankunding_core.models.text_classification.bert_UnifiedMC import BertUnifiedMC

from qiankunding_core.dataloaders.similarity.dataloader_UnifiedMC import TaskDataModelUnifiedMCForMatch
from qiankunding_core.models.similarity.bert_UnifiedMC import BertUnifiedMCForMatch

from qiankunding_core.dataloaders.nli.dataloader_UnifiedMC import TaskDataModelUnifiedMCForNLI
from qiankunding_core.models.nli.bert_UnifiedMC import BertUnifiedMCForNLI

# 设置gpu相关的全局变量
import qiankunding_core.utils.globalvar as globalvar
globalvar._init()
from qiankunding_core.utils.detect_gpu_memory import detect_gpu_memory, decide_gpu
gpu_memory, gpu_cur_used_memory = detect_gpu_memory()
globalvar.set_value("gpu_type", decide_gpu(gpu_memory))
globalvar.set_value('gpu_max_used_memory', gpu_cur_used_memory)

def download_model_from_huggingface(pretrained_model_dir, model_name, model_class=AutoModel, tokenizer_class=AutoTokenizer):
    if os.path.exists(os.path.join(pretrained_model_dir, model_name)):
        print("model already exists.")
        return
    cache_path = os.path.join(pretrained_model_dir, "cache")
    model = model_class.from_pretrained("IDEA-CCNL/" + model_name, cache_dir=cache_path)
    tokenizer = tokenizer_class.from_pretrained("IDEA-CCNL/" + model_name, cache_dir=cache_path)
    model.save_pretrained(os.path.join(pretrained_model_dir, model_name))
    tokenizer.save_pretrained(os.path.join(pretrained_model_dir, model_name))
    shutil.rmtree(cache_path)
    print("model %s is downloaded from huggingface." % model_name)

def generate_common_trainer(args, save_path):
    # Prepare Trainer
    checkpoint = ModelCheckpoint(dirpath=save_path,
                                    save_top_k=1,
                                    save_last=False,
                                    monitor='valid_acc_epoch',
                                    mode='max',
                                    filename='best_model')

    checkpoint.CHECKPOINT_NAME_LAST = "{epoch}-last"
    early_stop = EarlyStopping(monitor='valid_acc_epoch',
                                mode='max',
                                patience=10
                                ) 

    logger = loggers.TensorBoardLogger(save_dir=os.path.join(save_path, 'logs/'))
    trainer = Trainer.from_argparse_args(args, 
                                            logger=logger,
                                            callbacks=[checkpoint, early_stop])
    return trainer, checkpoint

def classification_pipeline(args):
    model_name = "Erlangshen-UniMC-MegatronBERT-1.3B-Chinese"
    # download pretrained model if not exists
    download_model_from_huggingface(args.pretrained_model_dir, model_name, model_class=MegatronBertForMaskedLM, tokenizer_class=BertTokenizer)
    # Set path to load pretrained model
    args.pretrained_model = os.path.join(args.pretrained_model_dir, model_name)
    # set knn datastore
    args.knn_datastore_data = args.train_data
    # init tokenizer
    tokenizer = get_train_tokenizer(args=args)            
    tokenizer.save_pretrained(args.save_path)
    # init label
    shutil.copyfile(os.path.join(args.data_dir, args.label_data), os.path.join(args.save_path, "label.json"))
    # init model
    data_model = TaskDataModelUnifiedMC(args, tokenizer)
    #加载模型
    model = BertUnifiedMC(args, tokenizer)
    trainer, checkpoint = generate_common_trainer(args, args.save_path)
    # training
    trainer.fit(model, data_model)
    #验证集效果最好的模型文件地址
    checkpoint_path = checkpoint.best_model_path

    # knn lm
    model = BertUnifiedMC.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
    model.cuda()
    model.eval()
    knn_best_hyper, knn_datastores = knn_utils.knn_augmentation(model, data_model, args.save_path)
    with open(os.path.join(args.save_path, "knn_best_hyper.json"), mode="w") as f:
        json.dump(knn_best_hyper, f, indent=4)
    with open(os.path.join(args.save_path, "knn_datastores.pkl"), mode="wb") as f:
        pickle.dump(knn_datastores, f)

    if args.test_data:
        output_save_path = os.path.join(args.save_path, 'predictions/')
        if not os.path.exists(output_save_path):
            os.makedirs(output_save_path)

        # Evaluation
        print("Load checkpoint from {}".format(checkpoint_path))
        model = BertUnifiedMC.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
        model.cuda()
        model.eval() 

        # evaluation(args, model, data_model, output_save_path, mode='test', data_set="test")
        evaluator = Evaluator(args, model, data_model, output_save_path)
        evaluator.evaluation(mode='test', data_set="test")
    

def similarity_pipeline(args):
    """ write a traininig pipeline and return the checkpoint path of the best model """
    model_name = "Erlangshen-UniMC-MegatronBERT-1.3B-Chinese"
    # download pretrained model if not exists
    download_model_from_huggingface(args.pretrained_model_dir, model_name, model_class=MegatronBertForMaskedLM, tokenizer_class=BertTokenizer)
    # Set path to load pretrained model
    args.pretrained_model = os.path.join(args.pretrained_model_dir, model_name)
    # init tokenizer
    tokenizer = get_train_tokenizer(args=args)            
    tokenizer.save_pretrained(args.save_path)
    # init label
    # shutil.copyfile(os.path.join(args.data_dir, args.label_data), os.path.join(args.save_path, "label.json"))
    # init model
    data_model = TaskDataModelUnifiedMCForMatch(args, tokenizer)
    #加载模型
    model = BertUnifiedMCForMatch(args, tokenizer)
    trainer, checkpoint = generate_common_trainer(args, args.save_path)
    # training
    trainer.fit(model, data_model)
    #验证集效果最好的模型文件地址
    checkpoint_path = checkpoint.best_model_path
    
    if args.test_data:
        output_save_path = os.path.join(args.save_path, 'predictions/')
        if not os.path.exists(output_save_path):
            os.makedirs(output_save_path)

        # Evaluation
        print("Load checkpoint from {}".format(checkpoint_path))
        model = BertUnifiedMCForMatch.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
        model.cuda()
        model.eval() 

        evaluator = SentencePairEvaluator(args, model, data_model, output_save_path)
        evaluator.evaluation(mode='test', data_set="test")
    # return None

def nli_pipeline(args):
    """ write a traininig pipeline and return the checkpoint path of the best model """
    model_name = "Erlangshen-UniMC-MegatronBERT-1.3B-Chinese"
    # download pretrained model if not exists
    download_model_from_huggingface(args.pretrained_model_dir, model_name, model_class=MegatronBertForMaskedLM, tokenizer_class=BertTokenizer)
    # Set path to load pretrained model
    args.pretrained_model = os.path.join(args.pretrained_model_dir, model_name)
    # init tokenizer
    tokenizer = get_train_tokenizer(args=args)            
    tokenizer.save_pretrained(args.save_path)
    # init label
    # shutil.copyfile(os.path.join(args.data_dir, args.label_data), os.path.join(args.save_path, "label.json"))
    # init model
    data_model = TaskDataModelUnifiedMCForNLI(args, tokenizer)
    #加载模型
    model = BertUnifiedMCForNLI(args, tokenizer)
    trainer, checkpoint = generate_common_trainer(args, args.save_path)
    # training
    trainer.fit(model, data_model)
    #验证集效果最好的模型文件地址
    checkpoint_path = checkpoint.best_model_path
    
    if args.test_data:
        output_save_path = os.path.join(args.save_path, 'predictions/')
        if not os.path.exists(output_save_path):
            os.makedirs(output_save_path)

        # Evaluation
        print("Load checkpoint from {}".format(checkpoint_path))
        model = BertUnifiedMCForNLI.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
        model.cuda()
        model.eval() 

        evaluator = SentencePairEvaluator(args, model, data_model, output_save_path)
        evaluator.evaluation(mode='test', data_set="test")


def train(args):
    # Set global random seed
    seed_everything(args.seed)
    
    # Set path to save checkpoint and outputs
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.output_dir = save_path


    # Save args
    with open(os.path.join(save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print('-' * 30 + 'Args' + '-' * 30)
    for k, v in vars(args).items():
        print(k, ":", v, end=',\t')
    print('\n' + '-' * 64)

    if args.task_type == "classification":
        classification_pipeline(args)
    elif args.task_type == "similarity":
        similarity_pipeline(args)
    elif args.task_type == "nli":
        nli_pipeline(args)

def main():
    total_parser = argparse.ArgumentParser()

    total_parser.add_argument("--task_dir", required=True, 
                            type=str, help="specific task directory")
    total_parser.add_argument("--task_type", required=True,
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
    # total_parser.add_argument('--unlabeled_data', default='unlabeled.json', type=str)   
    total_parser.add_argument('--save_path', default='output',
                            type=str, help="save path for trained model and other logs")

    # * Args for general setting
    total_parser.add_argument('--seed', default=1234,
                            type=int, help="random seed for training")
    total_parser.add_argument('--lr', default=2e-5,
                            type=float, help="learning rate")

    total_parser = Trainer.add_argparse_args(total_parser)
    print("total_parser:",total_parser)
    # * Args for data preprocessing
    args = total_parser.parse_args()

    print("pretrained_model_dir", args.pretrained_model_dir)
    args.gpus = 1
    args.num_sanity_val_steps = 1000 
    args.accumulate_grad_batches = 8 
    args.val_check_interval = 0.25 


    print('args', args)
    torch.set_num_threads(8)
    
    # main(args)
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
    