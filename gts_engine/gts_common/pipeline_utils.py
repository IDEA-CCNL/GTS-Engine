#encoding=utf8

import os
import json
import shutil
from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoModel, AutoTokenizer

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


def load_args(save_path):
    args_path = os.path.join(save_path, "args.json")
    print("Load args from {}".format(args_path))
    args_dict = json.load(open(args_path))
    args = ObjDict(args_dict)
    return args