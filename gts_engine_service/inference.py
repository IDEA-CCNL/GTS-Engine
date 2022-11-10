import os
import json
import pickle
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, BertTokenizer

# 设置gpu相关的全局变量
import qiankunding_core.utils.globalvar as globalvar
globalvar._init()

from qiankunding_core.dataloaders.text_classification.dataloader_UnifiedMC import TaskDatasetUnifiedMC
from qiankunding_core.models.text_classification.bert_UnifiedMC import BertUnifiedMC
from qiankunding_core.dataloaders.text_classification.dataloader_UnifiedMC import TaskDataModelUnifiedMC
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
    elif task_type == "similarity":
        return prepare_sentence_pair_inference(save_path)

def inference_samples(task_type, samples, inference_suite):
    if task_type == "classification":
        return classification_inference(samples, inference_suite)
    elif task_type == "similarity":
        return sentence_pair_inference(samples, inference_suite)

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
    data_model = TaskDataModelUnifiedMC(inference_suite["args"], inference_suite["tokenizer"])

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
        collate_fn=data_model.collate_fn, \
        batch_size=inference_suite["args"].train_batchsize)

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

def prepare_sentence_pair_inference(save_path):
    return None

def sentence_pair_inference(samples, inference_suite):
    return None
