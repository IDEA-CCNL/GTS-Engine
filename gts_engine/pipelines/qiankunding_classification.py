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
from qiankunding.utils.tokenization import get_train_tokenizer
from qiankunding.utils import knn_utils
from qiankunding.dataloaders.text_classification.dataloader_UnifiedMC import TaskDatasetUnifiedMC, TaskDataModelUnifiedMC, unifiedmc_collate_fn
from qiankunding.models.text_classification.bert_UnifiedMC import BertUnifiedMC
from qiankunding.dataloaders.text_classification.dataloader_tcbert import TaskDataModelTCBert
from qiankunding.models.text_classification.tcbert import TCBert
from qiankunding.utils.evaluation import Evaluator
from qiankunding.utils.knn_utils import knn_inference
from qiankunding.utils.utils import json2list, list2json
from gts_common.logs_utils import Logger

logger = Logger().get_log()


def train_classification(args):
    if args.train_mode == "standard":
        model_name = "Erlangshen-UniMC-MegatronBERT-1.3B-Chinese"
    elif args.train_mode == "advanced":
        model_name = "Erlangshen-TCBert-1.3B-Classification-Chinese"
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
    if args.train_mode == "standard":
        data_model = TaskDataModelUnifiedMC(args, tokenizer)
    elif args.train_mode == "advanced":
        data_model = TaskDataModelTCBert(args, tokenizer)
    #加载模型
    if args.train_mode == "standard":
        model = BertUnifiedMC(args, tokenizer)
    elif args.train_mode == "advanced":
        model = TCBert(args, tokenizer)
    trainer, checkpoint = generate_common_trainer(args, args.save_path)
    # training
    trainer.fit(model, data_model)
    #验证集效果最好的模型文件地址
    checkpoint_path = checkpoint.best_model_path

    # knn lm
    if args.train_mode == "standard":
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
            logger.info("Load checkpoint from {}".format(checkpoint_path))
            model = BertUnifiedMC.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
            model.cuda()
            model.eval() 

            # evaluation(args, model, data_model, output_save_path, mode='test', data_set="test")
            evaluator = Evaluator(args, model, data_model, output_save_path)
            test_acc = evaluator.evaluation(mode='test', data_set="test")

            task_info = json.load(open(os.path.join(args.task_dir, "task_info.json")))
            task_info["test_acc"] = test_acc
            with open(os.path.join(args.task_dir, "task_info.json"), mode="w") as f:
                    json.dump(task_info, f, indent=4)

    elif args.train_mode == "advanced":
        logger.info("Load checkpoint from {}".format(checkpoint_path))
        model = TCBert.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
        model.cuda()
        model.eval()
        output_save_path = os.path.join(args.save_path, 'predictions/')
        if not os.path.exists(output_save_path):
            os.makedirs(output_save_path)
        evaluator = Evaluator(args, model, data_model, output_save_path)
        test_acc = evaluator.evaluation(mode='test', data_set="unlabeled", threshold=args.threshold)


@PIPELINE_REGISTRY.register(suffix=__name__)
def train_pipeline(args):
    # save args
    args = save_args(args)
    if args.train_mode == "advanced":
        logger.info("******start advanced train******")
        train_classification(args)
        # shutil.rmtree(os.path.join(args.save_path, "best_model.ckpt"))
        os.remove(os.path.join(args.save_path, "best_model.ckpt"))
        pseudo_data = json2list(os.path.join(args.save_path, 'predictions','unlabeled_set_predictions.json'), use_key=["content", "label"])
        train_data = json2list(os.path.join(args.data_dir, args.train_data), use_key=["content", "label"])
        train_add_pseudo = train_data + pseudo_data
        list2json(train_add_pseudo, os.path.join(args.data_dir, "train_add_pseudo.json"), use_key=["content", "label"] )
        args.train_data = "train_add_pseudo.json"
        args.train_mode = "standard"
    logger.info("******start standard train******")
    train_classification(args)


@PIPELINE_REGISTRY.register(suffix=__name__)
def prepare_inference(save_path):
    # load args
    args = load_args(save_path)

    # load labels
    label_path = os.path.join(save_path, "label.json")
    line = json.load(open(label_path, 'r', encoding='utf8'))
    inference_choice = line['labels']

    # load tokenizer
    logger.info("Load tokenizer from {}".format(os.path.join(save_path, "vocab.txt")))
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

@PIPELINE_REGISTRY.register(suffix=__name__)
def inference(samples, inference_suite):
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