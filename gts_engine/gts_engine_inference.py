import os
import sys
import json
import pickle
import argparse
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, BertTokenizer

# 如果没有安装gts_engine，请把GTS-Engine/gts-engine加入到系统环境变量
sys.path.append(os.path.dirname(__file__))

# 设置gpu相关的全局变量
import qiankunding_core.utils.globalvar as globalvar
globalvar._init()

from qiankunding_core.dataloaders.text_classification.dataloader_UnifiedMC import TaskDatasetUnifiedMC
from qiankunding_core.dataloaders.nli.dataloader_UnifiedMC import TaskDatasetUnifiedMCForNLI,TaskDataModelUnifiedMCForNLI
from qiankunding_core.dataloaders.similarity.dataloader_UnifiedMC import TaskDatasetUnifiedMCForMatch,TaskDataModelUnifiedMCForMatch
from qiankunding_core.models.text_classification.bert_UnifiedMC import BertUnifiedMC
from qiankunding_core.dataloaders.text_classification.dataloader_UnifiedMC import unifiedmc_collate_fn
from qiankunding_core.utils.knn_utils import knn_inference


os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

# -----------------------------------推理模型的接口，按照任务类型调用------------------------------------------
def preprare_inference(task_type, save_path):
    if task_type == "classification":
        return prepare_classification_inference(save_path)
    elif task_type in ["similarity","nli"]:
        return prepare_sentence_pair_inference(task_type, save_path)

def inference_samples(task_type, samples, inference_suite):
    if task_type == "classification":
        return classification_inference(samples, inference_suite)
    elif task_type in ["similarity","nli"]:
        return sentence_pair_inference(task_type, samples, inference_suite)

# ------------------------------------------分类模型推理相关-------------------------------------------------
def prepare_classification_inference(save_path):
    # load args
    args = load_args(save_path)

    # load labels
    label_path = os.path.join(save_path, "label.json")
    line = json.load(open(label_path, 'r', encoding='utf8'))
    inference_choice = line['labels']

    # load tokenizer
    print("Load tokenizer from {}".format(os.path.join(save_path, "vocab.txt")))
    inference_tokenizer = BertTokenizer.from_pretrained(save_path)

    # load model
    checkpoint_path = os.path.join(save_path, "best_model.ckpt")
    inference_model = BertUnifiedMC.load_from_checkpoint(checkpoint_path, tokenizer=inference_tokenizer)
    inference_model.eval()
    inference_model = inference_model.cuda()

    # load knn data
    with open(os.path.join(save_path, "knn_best_hyper.json")) as f:
        knn_best_hyper = json.load(f)
    with open(os.path.join(save_path, "knn_datastores.pkl"), mode="rb") as f:
        knn_datastores = pickle.load(f)

    inference_suite = {
        "tokenizer": inference_tokenizer,
        "model": inference_model,
        "args": args,
        "choice": inference_choice,
        "knn_best_hyper": knn_best_hyper,
        "knn_datastores": knn_datastores
    }
    return inference_suite

def classification_inference(samples, inference_suite):
    # 加载数据
    inner_samples = []
    question = "请问下面的文字描述属于那个类别？"

    for idx, sample in enumerate(samples):
        inner_sample = {
            "id":idx,
            "content": sample["content"],
            "textb":"",
            "question":question,
            "choice":inference_suite["choice"],
            "label":inference_suite["choice"][0]
        }
        inner_samples.append(inner_sample)
    dataset = TaskDatasetUnifiedMC(
        data_path=None,
        args=inference_suite["args"],
        used_mask=False,
        tokenizer=inference_suite["tokenizer"],
        load_from_list=True,
        samples=inner_samples,
        choice=inference_suite["choice"]
    )
    
    dataloader = DataLoader(dataset, shuffle=False, 
        collate_fn=unifiedmc_collate_fn, \
        batch_size=inference_suite["args"].valid_batchsize)

    pred_labels = []
    pred_probs = []

    knn_best_hyper = inference_suite["knn_best_hyper"]
    knn_datastores = inference_suite["knn_datastores"]
    for batch in dataloader:
        logits, classify_probs, predicts, labels, sample_embeds = inference_suite["model"].predict(batch)
        knn_lambda = sum(knn_best_hyper["lambda_values"])
        final_probs = (1 - knn_lambda) * classify_probs
        knn_probs = knn_inference(sample_embeds, knn_datastores, knn_best_hyper)
        final_probs = final_probs + knn_lambda * knn_probs
        final_predicts = list(np.argmax(final_probs, axis=1))
        for predict, prob in zip(final_predicts, final_probs):    
            pred_labels.append(inference_suite["choice"][predict])
            pred_probs.append(prob.tolist())

    result = {'predictions':pred_labels, 'probabilities':pred_probs}
    return result

def prepare_sentence_pair_inference(task_type, save_path):
    # load args
    args = load_args(save_path)

    # load tokenizer
    print("Load tokenizer from {}".format(os.path.join(save_path, "vocab.txt")))
    inference_tokenizer = BertTokenizer.from_pretrained(save_path)

    # load model
    checkpoint_path = os.path.join(save_path, "best_model.ckpt")
    inference_model = BertUnifiedMC.load_from_checkpoint(checkpoint_path, tokenizer=inference_tokenizer)
    inference_model.eval()
    inference_model = inference_model.cuda()

    if task_type == "similarity":
        id2label = {0:'0', 1:'1'}
        data_model = TaskDataModelUnifiedMCForMatch(args, inference_tokenizer)
    elif task_type == "nli":
        id2label = {0:"entailment", 1:"contradiction", 2:"neutral"}
        data_model = TaskDataModelUnifiedMCForNLI(args, inference_tokenizer)
        

    inference_suite = {
        "tokenizer": inference_tokenizer,
        "model": inference_model,
        "data_model": data_model,
        "id2label": id2label,
        "args": args
    }
    return inference_suite

def sentence_pair_inference(task_type, samples, inference_suite):
    # 加载数据
    inner_samples = []
    question = "根据这段话"

    for idx, sample in enumerate(samples):
        textb = sample["sentence2"]
        label2id = {v:k for k,v in inference_suite["id2label"].items()}

        if task_type == "nli":
            choice = [f"可以推断出：{textb}",f"不能推断出：{textb}",f"很难推断出：{textb}"]  
        elif task_type == "similarity":
            choice = [f"不能理解为：{textb}",f"可以理解为：{textb}"]

        label = label2id[sample["label"]]
        inner_sample = {
            "id":idx,
            "texta": sample["sentence1"],
            "textb": sample["sentence2"],
            "question":question,
            "choice":choice,
            "label":label
        }
        inner_sample["answer"] = inner_sample["choice"][inner_sample["label"]]
        inner_samples.append(inner_sample)

    if task_type == "nli":
        dataset = TaskDatasetUnifiedMCForNLI(
            data_path=None,
            args=inference_suite["args"],
            used_mask=False,
            tokenizer=inference_suite["tokenizer"],
            load_from_list=True,
            samples=inner_samples,
            is_test=True,
            unlabeled=True
        )
    else:
        dataset = TaskDatasetUnifiedMCForMatch(
            data_path=None,
            args=inference_suite["args"],
            used_mask=False,
            tokenizer=inference_suite["tokenizer"],
            load_from_list=True,
            samples=inner_samples,
            is_test=True,
            unlabeled=True
        )

    
    dataloader = DataLoader(dataset, shuffle=False, 
        collate_fn=inference_suite["data_model"].collate_fn, \
        batch_size=inference_suite["args"].valid_batchsize)

    pred_labels = []
    pred_probs = []


    for batch in dataloader:
        logits, classify_probs, predicts, labels, sample_embeds = inference_suite["model"].predict(batch)

        final_predicts = list(np.argmax(classify_probs, axis=1))
        for predict, prob in zip(final_predicts, classify_probs):    
            pred_labels.append(inference_suite["id2label"][predict])
            pred_probs.append(prob.tolist())


    result = {'predictions':pred_labels, 'probabilities':pred_probs}
    return result

def main():
    total_parser = argparse.ArgumentParser()

    total_parser.add_argument("--task_dir", required=True, 
                            type=str, help="specific task directory")
    total_parser.add_argument("--task_type", required=True,
                            type=str, help="task type for training")
    total_parser.add_argument("--input_path", required=True,
                            type=str, help="input path of data which will be inferenced")
    total_parser.add_argument("--output_path", required=True,
                            type=str, help="output path of inferenced data")
    
    args = total_parser.parse_args()                            

    save_path = os.path.join(args.task_dir, "outputs")
    samples = []
    for line in open(args.input_path):
        line = line.strip()
        sample = json.loads(line)
        samples.append(sample)
    if args.task_type in ["nli","similarity"]:
        inference_suite = prepare_sentence_pair_inference(args.task_type, save_path)
        result = sentence_pair_inference(args.task_type, samples, inference_suite)
    else:
        inference_suite = preprare_inference(args.task_type, save_path)
        result = classification_inference(samples, inference_suite)

    with open(args.output_path, encoding="utf8", mode="w") as fout:
        json.dump(result, fout, ensure_ascii=False)

if __name__ == '__main__':    
    main()
    