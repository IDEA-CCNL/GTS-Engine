#encoding=utf8
import os
import sys
import json
import pickle
import shutil
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, MegatronBertForMaskedLM

from gts_common.registry import PIPELINE_REGISTRY
from gts_common.pipeline_utils import download_model_from_huggingface, generate_common_trainer, load_args, save_args
from qiankunding.utils.tokenization import get_t5_tokenizer
from qiankunding.utils import knn_utils
from qiankunding.dataloaders.text_generation.dataloader_kgt5 import TaskDataModelKGT5, TaskDatasetKGT5, kg_collate_fn
from qiankunding.models.text_generation.t5_kg import T5KG
from qiankunding.utils.evaluation import TextGenerateEvaluator
from qiankunding.utils.utils import json2list, list2json
from gts_common.logs_utils import Logger
from transformers import T5ForConditionalGeneration, BertTokenizer, T5Tokenizer

logger = Logger().get_log()


def train_generation(args):
    model_name = "Randeng-T5-Keyphrase-Generation-Sci"
    # download pretrained model if not exists
    download_model_from_huggingface(args.pretrained_model_dir, model_name, model_class=T5ForConditionalGeneration, tokenizer_class=T5Tokenizer)
    # Set path to load pretrained model
    args.pretrained_model = os.path.join(args.pretrained_model_dir, model_name)
    # init tokenizer
    tokenizer = get_t5_tokenizer(args=args)            
    tokenizer.save_pretrained(args.save_path)
    # init model
    data_model = TaskDataModelKGT5(args, tokenizer)
    #加载模型
    model = T5KG(args, tokenizer)
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
        logger.info("Load checkpoint from {}".format(checkpoint_path))
        model = T5KG.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
        model.cuda()
        model.eval() 

        evaluator = TextGenerateEvaluator(args, model, data_model, output_save_path)
        test_f1 = evaluator.evaluation(mode='test', data_set="test")

        task_info = json.load(open(os.path.join(args.task_dir, "task_info.json")))
        task_info["test_f1"] = test_f1
        with open(os.path.join(args.task_dir, "task_info.json"), mode="w") as f:
                json.dump(task_info, f, indent=4)


@PIPELINE_REGISTRY.register(suffix=__name__)
def train_pipeline(args):
    # save args
    args = save_args(args)
    logger.info("******start standard train******")
    train_generation(args)


@PIPELINE_REGISTRY.register(suffix=__name__)
def prepare_inference(save_path):
    # load args
    args = load_args(save_path)

    # load tokenizer
    logger.info("Load tokenizer from {}".format(os.path.join(save_path, "vocab.txt")))
    inference_tokenizer = T5Tokenizer.from_pretrained(save_path)

    # load model
    checkpoint_path = os.path.join(save_path, "best_model.ckpt")
    inference_model = T5KG.load_from_checkpoint(checkpoint_path, tokenizer=inference_tokenizer)
    inference_model.eval()
    inference_model = inference_model.cuda()

    inference_suite = {
        "tokenizer": inference_tokenizer,
        "model": inference_model,
        "args": args
    }
    return inference_suite

@PIPELINE_REGISTRY.register(suffix=__name__)
def inference(samples, inference_suite):
    # 加载数据
    inner_samples = []
    question = "请问下面的文字描述属于那个类别？"

    for idx, sample in enumerate(samples):
        inner_sample = {
            "id":idx,
            "content": sample["content"],
            "label":sample["label"],
        }
        inner_samples.append(inner_sample)
        
    dataset = TaskDatasetKGT5(
        data_path=None,
        args=inference_suite["args"],
        tokenizer=inference_suite["tokenizer"],
        load_from_list=True,
        samples=inner_samples
    )
    
    dataloader = DataLoader(dataset, shuffle=False, 
        collate_fn=kg_collate_fn, \
        batch_size=inference_suite["args"].valid_batchsize)

    pred_labels = []
    
    for batch in dataloader:
        _, _, predicts, labels = inference_suite["model"].predict(batch)
        
        for predict in predicts:
            pred_labels.append(predict)

    result = {'predictions':pred_labels}
    return result