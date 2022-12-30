#encoding=utf8

import os
import json
import shutil
from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoModel, AutoTokenizer
from gts_common.arguments import GtsEngineArgs
from gts_common.logs_utils import Logger
logger = Logger().get_log()

def download_model_from_huggingface(pretrained_model_dir, model_name, model_class=AutoModel, tokenizer_class=AutoTokenizer):
    if os.path.exists(os.path.join(pretrained_model_dir, model_name)):
        logger.info("model already exists.")
        return
    cache_path = os.path.join(pretrained_model_dir, model_name, "cache")
    model = model_class.from_pretrained("IDEA-CCNL/" + model_name, cache_dir=cache_path)
    tokenizer = tokenizer_class.from_pretrained("IDEA-CCNL/" + model_name, cache_dir=cache_path)
    model.save_pretrained(os.path.join(pretrained_model_dir, model_name))
    tokenizer.save_pretrained(os.path.join(pretrained_model_dir, model_name))
    shutil.rmtree(cache_path)
    logger.info("model %s is downloaded from huggingface." % model_name)

def generate_common_trainer(args: GtsEngineArgs, save_path):
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
    trainer = Trainer(
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        accumulate_grad_batches=8,
        val_check_interval=args.val_check_interval,
        num_sanity_val_steps=1000,
        gpus=args.gpus,
        logger=logger,
        callbacks=[checkpoint, early_stop]
    )
    return trainer, checkpoint

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


def save_args(args: GtsEngineArgs):
    args_path = os.path.join(args.save_path, "args.json")
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logger.info("Save args to {}".format(args_path))
    for k, v in vars(args).items():
        #logger.info(k, ":", v,',\t')
        logger.info("{} : {}    ".format(k,v))
    logger.info('\n' + '-' * 64)
    return args


def load_args(save_path):
    args_path = os.path.join(save_path, "args.json")
    logger.info("Load args from {}".format(args_path))
    args_dict = json.load(open(args_path))
    args = ObjDict(args_dict)
    return args