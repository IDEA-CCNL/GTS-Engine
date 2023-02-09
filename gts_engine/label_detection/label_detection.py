import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime

from gts_common.logs_utils import Logger
from gts_common.pipeline_utils import download_model_from_huggingface
from qiankunding.dataloaders.text_classification.dataloader_UnifiedMC import \
    TaskDatasetUnifiedMC
from qiankunding.models.nli.bert_UnifiedMC import taskModel
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, MegatronBertForMaskedLM

logger = Logger().get_log()


def read_data(label_path, data_path):
    input_data = []
    with open(os.path.join(data_path), encoding='utf8') as f:
        for line in f:
            input_data.append(json.loads(line.strip()))

    input_labels = object
    with open(os.path.join(label_path), encoding='utf8') as f:
        input_labels = json.load(f)
    return input_data, input_labels


def take_sample_input(
    labels_length,
    input_data,
):
    labels_count = labels_length
    sample_count_every_class = int(2000 / labels_count)
    logger.info(f"sample_count_every_class {sample_count_every_class}")
    random.shuffle(input_data)
    sample_input_data = []
    label_to_count = {}
    for data in input_data:
        label_to_count[data['id']] = label_to_count.get(data['id'], 0) + 1
        if label_to_count[data['id']] <= sample_count_every_class:
            sample_input_data.append(data)
    return sample_input_data


def take_sample(sample_input_data, current_label):
    # 样本采样  1:2 正负样本
    total_sentences = [i['content'] for i in sample_input_data]
    total_val_data = sample_input_data
    total_val_data_index = [i for i in range(len(total_val_data))]
    true_label_index = [
        i for i, k in enumerate(total_val_data) if k['label'] == current_label
    ]

    tmp_set = set(total_val_data_index).difference(set(true_label_index))
    # 负采样个数
    negative_label_index = random.sample(
        list(tmp_set), min(len(true_label_index) * 2, len(list(tmp_set))))
    need_index = true_label_index + negative_label_index
    need_index.sort()

    sentences = []
    val_data = []
    for i in need_index:
        sentences.append(total_sentences[i])
        val_data.append(total_val_data[i])
    return sentences, val_data


def predict_process(two_choices, tokenizer, sentences, model):
    logger.info(f"start {two_choices}")
    samples = []
    for sentence in sentences:
        tmp_sample = {"content": sentence, "label": two_choices[0]}
        samples.append(tmp_sample)

    train_data = TaskDatasetUnifiedMC(data_path=None,
                                      args=args,
                                      used_mask=False,
                                      tokenizer=tokenizer,
                                      load_from_list=True,
                                      samples=samples,
                                      choice=two_choices)

    train_dataloader = DataLoader(train_data,
                                  shuffle=False,
                                  batch_size=1,
                                  pin_memory=False)

    total_predicts = []

    for batch in tqdm(train_dataloader):

        inputs = {
            'input_ids': batch['input_ids'].cuda(),
            'attention_mask': batch['attention_mask'].cuda(),
            'token_type_ids': batch['token_type_ids'].cuda(),
            "position_ids": batch['position_ids'].cuda(),
            "mlmlabels": batch['mlmlabels'].cuda(),
            "clslabels": batch['clslabels'].cuda(),
            "clslabels_mask": batch['clslabels_mask'].cuda(),
            "mlmlabels_mask": batch['mlmlabels_mask'].cuda(),
        }

        loss, logits, cls_logits, hidden_states = model(**inputs)

        probs = torch.nn.functional.softmax(cls_logits, dim=-1)
        predicts = torch.argmax(probs, dim=-1)
        score = torch.max(probs, dim=-1)[0]

        probs = probs.detach().cpu().numpy()
        predicts = predicts.detach().cpu().numpy()
        score = score.detach().cpu().numpy()

        label_idx = list(batch["label_idx"][0].numpy())
        total_predicts += [label_idx.index(i) for i in predicts]
    return total_predicts


def label_detection(label_path, data_path):
    model_name = "Erlangshen-UniMC-MegatronBERT-1.3B-Chinese"
    pretrained_model_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "pretrained")
    download_model_from_huggingface(pretrained_model_dir,
                                    model_name,
                                    model_class=MegatronBertForMaskedLM,
                                    tokenizer_class=BertTokenizer)
    model_path = os.path.join(pretrained_model_dir, model_name)
    starttime = datetime.now()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = taskModel(model_path, tokenizer=tokenizer)
    model.eval()
    model.cuda()
    random.seed(123)
    labels_info = {}
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'label_detection_result.json')
    input_data, input_labels = read_data(label_path=label_path,
                                         data_path=data_path)
    sample_input_data = take_sample_input(len(input_labels["labels"]),
                                          input_data)
    label_detection_result = []

    total_count = 0
    index = 0
    for label_value in input_labels["labels"]:
        current_label = label_value
        current_label_id = index
        index += 1

        sentences, val_data = take_sample(sample_input_data=sample_input_data,
                                          current_label=current_label)

        total_count += len(sentences)
        random_f1 = -1
        label_f1 = -1
        two_choices = [
            '[MASK]' * len(current_label), '[MASK]' * len(current_label)
        ]
        n = 0
        while n < 2:
            total_predicts = predict_process(two_choices=two_choices,
                                             tokenizer=tokenizer,
                                             sentences=sentences,
                                             model=model)

            y_true = []
            y_pred = []
            y_senetence = []
            for sample, predict in zip(val_data, total_predicts):
                if sample['label'] == current_label:
                    y_true.append(0)
                else:
                    y_true.append(1)
                y_pred.append(predict)
                if predict == 0:
                    y_senetence.append(sample['content'])

            trueY = np.array(y_true)
            testY = np.array(y_pred)
            logger.info(f"n {n}")

            label = 0

            if n == 0:
                random_f1 = f1_score(trueY == label,
                                     testY == label,
                                     labels=True)
                random_recall = recall_score(trueY == label,
                                             testY == label,
                                             labels=True)
            else:
                label_f1 = f1_score(trueY == label,
                                    testY == label,
                                    labels=True)
                label_recall = recall_score(trueY == label,
                                            testY == label,
                                            labels=True)

            n += 1
            two_choices[0] = current_label

        f1_info = []
        if label_f1 > random_f1 * 1.1:
            labels_info[current_label] = ["2", current_label_id, label_f1]
            f1_info.append((label_f1, random_f1))

        elif label_f1 <= random_f1 * 1.1 and label_f1 > random_f1:
            if label_recall > random_recall:
                labels_info[current_label] = ["2", current_label_id, label_f1]
            else:
                labels_info[current_label] = ["1", current_label_id, label_f1]
            f1_info.append((label_f1, random_f1))

        elif label_f1 <= random_f1:
            if label_recall > random_recall:
                labels_info[current_label] = ["1", current_label_id, label_f1]
            else:
                labels_info[current_label] = ["0", current_label_id, label_f1]
            f1_info.append((label_f1, random_f1))

        label_detection_result.append(
            (current_label, labels_info[current_label], f1_info))

    tmp_result = []
    index = 0
    for label_value in input_labels["labels"]:
        if label_value in labels_info:
            current_label = label_value
            tmp_result.append(
                (current_label, labels_info[current_label][0],
                 labels_info[current_label][1], labels_info[current_label][2]))
        index += 1
    tmp_result.sort(key=lambda x: (x[1], x[3]))

    result = []
    for label, rank, label_id, score in tmp_result:
        result.append({
            "label": label,
            "rank": rank,
            "label_id": label_id,
            "score": score
        })

    logger.info(f"label_detection_result {label_detection_result}")
    logger.info(f"result {result}")
    endtime = datetime.now()
    logger.info("RunTime: {}h-{}m-{}s".format(
        endtime.hour - starttime.hour, endtime.minute - starttime.minute,
        endtime.second - starttime.second))

    with open(result_path, 'w', encoding='utf-8') as f:
        for i in result:
            f.write(json.dumps(i, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    total_parser = argparse.ArgumentParser()
    total_parser.add_argument("--label_path", type=str, help="data path")
    total_parser.add_argument("--data_path", type=str, help="data path")
    total_parser.add_argument("--max_len", default=630, type=str)
    args = total_parser.parse_args()

    logger.info(f"args: {args}")

    label_detection(label_path=args.label_path, data_path=args.data_path)
