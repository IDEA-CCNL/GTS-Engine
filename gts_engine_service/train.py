import os
import time 
import json
import torch
import shutil
import argparse
import traceback
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from transformers import AutoModel, AutoTokenizer, BertTokenizer, MegatronBertForMaskedLM, MegatronBertConfig

from qiankunding_core.utils.evaluation import evaluation
from qiankunding_core.utils.tokenization import get_train_tokenizer

from qiankunding_core.utils import knn_utils

from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from qiankunding_core.dataloaders.text_classification.dataloader_UnifiedMC import TaskDataModelUnifiedMC
from qiankunding_core.models.text_classification.bert_UnifiedMC import BertUnifiedMC

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

def generate_common_trainer(save_path):
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
                                patience=10,
                            #    check_on_train_epoch_end=True # Check early stopping after every train epoch, ignore multi validation in one train epoch
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
    # init tokenizer
    tokenizer = get_train_tokenizer(args=args)            
    tokenizer.save_pretrained(args.save_path)
    # init label
    shutil.copyfile(os.path.join(args.data_dir, args.label_data), os.path.join(args.save_path, "label.json"))
    # init model
    data_model = TaskDataModelUnifiedMC(args, tokenizer)
    #加载模型
    model = BertUnifiedMC(args, tokenizer)
    trainer, checkpoint = generate_common_trainer(args.save_path)
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
        model = BertUnifiedMC.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
        model.cuda()
        model.eval() 

        evaluation(args, model, data_model, output_save_path, mode='test', data_set="test")
    

def sentence_pair_pipeline(args):
    """ write a traininig pipeline and return the checkpoint path of the best model """
    # TODO
    return None


def main(args):
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
        sentence_pair_pipeline(args)
    elif args.task_type == "nli":
        sentence_pair_pipeline(args)



if __name__ == '__main__':    
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

    total_parser = argparse.ArgumentParser()

    total_parser.add_argument("--task_dir", required=True, 
                            type=str, help="train task dir")
    total_parser.add_argument("--task_type", required=True,
                            type=str, help="task type for training")
    total_parser.add_argument("--use_knn", default=False, action="store_true",
                            help="whether or not to use knn component")
    
    total_parser.add_argument('--num_workers', default=8, type=int)
    
    
    total_parser.add_argument('--train_batchsize', default=1, type=int)   
    total_parser.add_argument('--valid_batchsize', default=4, type=int)   
    total_parser.add_argument('--test_batchsize', default=4, type=int)  
    total_parser.add_argument('--max_len', default=512, type=int)   
    total_parser.add_argument('--recreate_dataset', action='store_true', default=True)

    total_parser.add_argument('--data_dir',default='files/data',type=str)
    total_parser.add_argument('--output_dir',default='',type=str)
    total_parser.add_argument('--train_data', default='train.json', type=str)   
    total_parser.add_argument('--valid_data', default='dev.json', type=str)     
    total_parser.add_argument('--test_data', default='test.json', type=str)      
    total_parser.add_argument('--label_data', default='labels.json', type=str)
    total_parser.add_argument('--unlabeled_data', default='unlabeled.json', type=str)   
    total_parser.add_argument('--label2id_file', default=None, type=str)  
    total_parser.add_argument('--content_key', default="content",help="content key in json file")
    total_parser.add_argument('--label_key', default="label",help="label key in json file")
    total_parser.add_argument('--pseudo_labeling', action='store_true', default=False,help="whether to do psedudo labeling in unlabeled data")
    total_parser.add_argument('--do_masking', action='store_true', default=False,help="")
    total_parser.add_argument('--do_wwm', action='store_true', default=False,help="")
    total_parser.add_argument('--wwm_mask_rate', default=0.12,help="")
    total_parser.add_argument('--do_label_guide', action='store_true', default=False,help="")
    total_parser.add_argument('--label_guided_rate', default=0.5,help="")

    total_parser.add_argument('--output',default='output',type=str)
    total_parser.add_argument('--save_path', default='output', type=str)
    # total_parser.add_argument('--model_path',default='{}/model_save'.format(os.getcwd()),type=str)

    # * Args for general setting
    total_parser.add_argument('--num_threads', default=8, type=int)
    total_parser.add_argument('--eval', action='store_true', default=False)
    total_parser.add_argument('--checkpoint_path', default=None, type=str)
    total_parser.add_argument('--seed', default=1234, type=int)
    total_parser.add_argument('--save_dir', default='./save', type=str)
    total_parser.add_argument('--model_name', default='megatron_bert', type=str)
    total_parser.add_argument('--lr', default=2e-5, type=float)
    total_parser.add_argument('--l2', default=0., type=float)
    total_parser.add_argument('--warmup', default=0.1, type=float)
    total_parser.add_argument('--adv', action='store_true', default=False, help="if use adversarial training")
    total_parser.add_argument('--divergence', default='js', type=str)
    total_parser.add_argument('--adv_nloop', default=1, type=int,
                        help="1 (default), inner loop for getting the best perturbations.")
    total_parser.add_argument('--adv_step_size', default=1e-3, type=float,
                        help="1 (default), perturbation size for adversarial training.")
    total_parser.add_argument('--adv_alpha', default=1, type=float,
                        help="1 (default), trade off parameter for adversarial training.")
    total_parser.add_argument('--noise_var', default=1e-5, type=float)
    total_parser.add_argument('--noise_gamma', default=1e-6, type=float, help="1e-4 (default), eps for adversarial copy training.")
    total_parser.add_argument('--project_norm_type', default='inf', type=str)
    total_parser.add_argument('--nlabels', default=10, type=int)

    # * Args for base specific model 
    total_parser.add_argument("--pretrained_model_dir", default="",
                        type=str, help="Path to the directory which contains all the pretrained models downloaded from huggingface")
    total_parser.add_argument('--child_tuning_p', type=float, default=1.0, help="prob of dropout gradient, if < 1.0, use child-tuning")
    total_parser.add_argument('--finetune', action='store_true', default=True, help="if fine tune the pretrained model")    #####
    total_parser.add_argument("--pooler_type", type=str, default="cls_pooler", help="acceptable values:[cls, cls_before_pooler, avg, avg_top2, avg_first_last]")
    total_parser.add_argument('--bert_lr', default=2e-5, type=float)
    total_parser.add_argument('--bert_l2', default=0., type=float)
    total_parser.add_argument('--mlp_dropout', default=0.5, type=float, help="Dropout rate in MLP layer")
    total_parser.add_argument('--load_from_tapt', action='store_true', default=False, help="Dropout rate in MLP layer")
    #total_parser = Bert.add_model_specific_args(total_parser)


    total_parser = Trainer.add_argparse_args(total_parser)
    print("total_parser:",total_parser)
    # * Args for data preprocessing
    args = total_parser.parse_args()

    args.pretrained_model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pretrained")
    args.gpus = 1
    args.num_sanity_val_steps = 1000 
    args.accumulate_grad_batches = 8 
    args.warmup = 0.1 
    args.num_threads = 8 
    args.val_check_interval = 0.25 


    print('args', args)
    torch.set_num_threads(args.num_threads)
    
    task_info_path = os.path.join(args.task_dir, "task_info.json")
    task_info = json.load(open(task_info_path))

    try:
        main(args)
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
        


    
    