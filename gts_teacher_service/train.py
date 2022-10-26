

import os
import json
import string
import torch
import argparse
from itertools import chain
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from transformers import AutoModel, AutoTokenizer, AdamW, BertTokenizer

from teacher_core.dataloaders.text_classification.dataloader import TaskDataset, TaskDataModel
#from teacher_core.models.text_classification.bert_baseline import Bert

from teacher_core.utils.evaluation import evaluation
from teacher_core.utils.tokenization import get_train_tokenizer

from teacher_core.utils import knn_utils

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, loggers
# from pytorch_lightning.callbacks.progress import tqdm
from tqdm.auto import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from teacher_config import tuning_methods_config

from teacher_core.dataloaders.text_classification.dataloader_UnifiedMC import TaskDataModelUnifiedMC


from teacher_core.models.text_classification.bert_UnifiedMC import BertUnifiedMC

def main(args):
    # Set global random seed
    seed_everything(args.seed)
    # Set path to save checkpoint and outputs
    
    
    # hyparas = 'tuning_method={}-seed={}-model={}-bs={}-train_data={}-lr={}-pooler={}-l2={}-ft={}-clip={}-drop={}-adv={}-prec-{}'.format(
    #             args.tuning_method, args.seed,args.pretrained_model_name,args.train_batchsize, args.train_data,args.lr, args.pooler_type, args.l2,
    #             int(args.finetune), args.gradient_clip_val, args.mlp_dropout, int(args.adv), args.precision)
    
    # save_path = os.path.join(args.save_dir, args.output)
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # args.save_path = save_path
    args.output_dir = save_path
    
    # Prepare Trainer
    checkpoint = ModelCheckpoint(dirpath=save_path,
                                    save_top_k=1,
                                    save_last=True,
                                    monitor='valid_acc_epoch',
                                    mode='max',
                                    filename='{epoch:02d}-{valid_acc:.4f}')

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
    # Set path to load pretrained model
    args.pretrained_model = os.path.join(args.pretrained_model_dir, args.pretrained_model_name)

    # Save args
    with open(os.path.join(save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print('-' * 30 + 'Args' + '-' * 30)
    for k, v in vars(args).items():
        print(k, ":", v, end=',\t')
    print('\n' + '-' * 64)

    
    tokenizer = get_train_tokenizer(args=args)            
    tokenizer.save_pretrained(save_path)
    # data_model = TaskDataModel(args, tokenizer)
    # model = Bert(args, tokenizer)


    # data_model = tuning_methods_config[args.tuning_method]["DataModel"](args, tokenizer)
    # model = tuning_methods_config[args.tuning_method]["TuningModel"](args, tokenizer)


    data_model = TaskDataModelUnifiedMC(args, tokenizer)
    model = BertUnifiedMC(args, tokenizer)
    for k,v in model.named_parameters():
        print('named_parameters',  k,v.requires_grad)
    trainer.fit(model, data_model)
    checkpoint_path = checkpoint.best_model_path

    

    
    # # Evaluating on dev set
    # output_save_path = os.path.join(save_path, 'predictions/')
    # if not os.path.exists(output_save_path):
    #     os.makedirs(output_save_path)

    if args.test_data:
        output_save_path = os.path.join(save_path, 'predictions/')
        if not os.path.exists(output_save_path):
            os.makedirs(output_save_path)

        # Evaluation
        print("Load checkpoint from {}".format(checkpoint_path))
        model = BertUnifiedMC.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
        model.cuda()
        model.eval() 

        evaluation(args, model, data_model, output_save_path, mode='test', data_set="test")






if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    

    total_parser = argparse.ArgumentParser()

    print("total_parser v1:",total_parser)
    total_parser.add_argument("--task_name", default="tnews", 
                            type=str, help="train task name")     #####
    # total_parser.add_argument("--tuning_method", default="UnifiedMC",   #####
    #                         type=str, help="tuning method")
    total_parser.add_argument("--use_knn", default=False, action="store_true",
                            help="whether or not to use knn component")
    
    total_parser.add_argument('--num_workers', default=8, type=int)
    
    
    total_parser.add_argument('--train_batchsize', default=1, type=int)    ###
    total_parser.add_argument('--valid_batchsize', default=4, type=int)    #####
    total_parser.add_argument('--max_len', default=512, type=int)    #####
    total_parser.add_argument('--recreate_dataset', action='store_true', default=True)

    total_parser.add_argument('--data_dir',default='./data',type=str)
    total_parser.add_argument('--output_dir',default='',type=str)
    total_parser.add_argument('--train_data', default='train_0.json', type=str)   ######
    total_parser.add_argument('--valid_data', default='dev_0.json', type=str)     #######
    total_parser.add_argument('--test_data', default='test_public.json', type=str)   ######
    total_parser.add_argument('--unlabeled_data', default='unlabeled.json', type=str)   ######
    total_parser.add_argument('--label2id_file', default=None, type=str)
    total_parser.add_argument('--content_key', default="content",help="content key in json file")
    total_parser.add_argument('--label_key', default="label",help="label key in json file")
    total_parser.add_argument('--pseudo_labeling', action='store_true', default=False,help="whether to do psedudo labeling in unlabeled data")
    total_parser.add_argument('--do_masking', action='store_true', default=False,help="")
    total_parser.add_argument('--do_wwm', action='store_true', default=False,help="")
    total_parser.add_argument('--wwm_mask_rate', default=0.12,help="")
    total_parser.add_argument('--do_label_guide', action='store_true', default=False,help="")
    total_parser.add_argument('--label_guided_rate', default=0.5,help="")

    total_parser.add_argument('--sup_pretrain', action='store_true', default=False,help="UGC预训练")

    total_parser.add_argument('--output',default='output',type=str)
    # total_parser.add_argument('--model_path',default='{}/model_save'.format(os.getcwd()),type=str)
    total_parser.add_argument('--save_path',default='{}/save'.format(os.getcwd()),type=str)


    
    # * Args for base model 
        

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

    
    total_parser.add_argument("--pretrained_model_dir", default="{}".format(os.path.abspath(os.path.join(os.getcwd(), ".."))),    #######
                        type=str, help="Path to the directory which contains all the pretrained models downloaded from huggingface")
    total_parser.add_argument('--pretrained_model_name',
                        default='UnifiedMC_Bert-1.3B',   #######
                        type=str)
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

    args.gpus = 1
    args.max_epochs = 1 
    args.min_epochs = 1 
    args.num_threads = 8 
    args.seed = 123 
    args.val_check_interval = 0.25 
    args.num_sanity_val_steps = 1000 
    args.accumulate_grad_batches = 8 
    args.warmup = 0.1 

    # args.data_dir='/raid/liuyibo/tnewsf'

    print('args', args)
    torch.set_num_threads(args.num_threads)
    
    
    
    with open(os.path.join(os.getcwd(),'info.json'),'w') as f:
        f.write(json.dumps({"训练状态":"训练中"},ensure_ascii=False))


    try:
        main(args)
        with open(os.path.join(os.getcwd(),'info.json'),'w') as f:
            f.write(json.dumps({"训练状态":"训练完成"},ensure_ascii=False))
    except:
        with open(os.path.join(os.getcwd(),'info.json'),'w') as f:
            f.write(json.dumps({"训练状态":"训练失败"},ensure_ascii=False))


    
    