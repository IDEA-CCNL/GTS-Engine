import os
import time
import argparse
import numpy as np
from qiankunding.dataloaders.text_classification.dataloader_UnifiedMC import TaskDatasetUnifiedMC
from qiankunding.models.nli.bert_UnifiedMC import taskModel
from gts_common.logs_utils import Logger
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import random
from sklearn.metrics import precision_score, f1_score, recall_score
from datetime import datetime


logger = Logger().get_log()


def label_detection(model_path, data_path):
    start = time.time()
    starttime = datetime.now()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = taskModel(model_path, tokenizer=tokenizer)
    model.eval()
    model.cuda()
    random.seed(123)

    input_data = []
    with open(os.path.join(data_path, 'tnews_test.json'), 'r',
              encoding='utf8') as f:
        for line in f:
            input_data.append(json.loads(line.strip()))

    input_labels = object
    with open(os.path.join(data_path, 'tnews_label.json'),
              'r',
              encoding='utf8') as f:
        input_labels = json.load(f)

    # 每类对应数据采样  避免数据过多导致预测时间过长
    labels_count = len(input_labels["labels"])
    # sample_count_every_class = int(6000/labels_count)   # 600/(0.05 * 2)
    sample_count_every_class = int(2000 / labels_count)  # 600/(0.05 * 2)
    logger.info("sample_count_every_class {}".format(sample_count_every_class))
    random.shuffle(input_data)
    sample_input_data = []
    label_to_count = {}
    for data in input_data:
        label_to_count[data['id']] = label_to_count.get(data['id'], 0) + 1
        if label_to_count[data['id']] <= sample_count_every_class:
            sample_input_data.append(data)

    labels_info = {}
    result_path = os.path.join(data_path, 'result.json')
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf8') as f:
            for line in f:
                line = json.loads(line.strip())
                labels_info[line['label']] = [
                    line['rank'], line['label_id'], line['score']
                ]

    # question = "请问下面的文字描述属于那个类别？"
    label_detection_result = []

    total_count = 0
    index = 0
    for value in input_labels["labels"]:
        # if input_label["current_label"] == input_label["last_label"]:    #不一致需要检测
        #    continue
        # if input_label["status"] == 0:  # 不一致需要检测
        #    continue
        current_label = value
        current_label_id = index
        index += 1

        # 样本采样  1:2 正负样本
        total_sentences = [i['content'] for i in sample_input_data]
        total_vail_data = sample_input_data
        total_vail_data_index = [i for i in range(len(total_vail_data))]
        true_label_index = [
            i for i, k in enumerate(total_vail_data)
            if k['label'] == current_label
        ]

        tmp_set = set(total_vail_data_index).difference(set(true_label_index))
        negative_label_index = list(tmp_set)
        random.shuffle(negative_label_index)
        # 负采样个数
        negative_label_index = negative_label_index[:len(true_label_index) * 2]
        need_index = true_label_index + negative_label_index
        need_index.sort()

        sentences = []
        vail_data = []
        for i in need_index:
            sentences.append(total_sentences[i])
            vail_data.append(total_vail_data[i])

        total_count += len(sentences)
        random_f1 = -1
        label_f1 = -1
        two_choices = [
            '[MASK]' * len(current_label), '[MASK]' * len(current_label)
        ]
        # two_choices[1] = random.choices(choices)[0]
        n = 0
        while n < 2:
            logger.info("start {}".format(two_choices))
            samples = []
            for sentence in sentences:
                tmp_sample = {
                    "content": sentence,
                    # "textb": "",
                    # "question": question,
                    # "choice": two_choices,
                    "label": two_choices[0]
                }
                samples.append(tmp_sample)

            start1 = time.time()
            train_data = TaskDatasetUnifiedMC(data_path=None,
                                              args=args,
                                              used_mask=False,
                                              tokenizer=tokenizer,
                                              load_from_list=True,
                                              samples=samples,
                                              choice=two_choices)
            # (self, data_path=None, args=None,used_mask=True, tokenizer=None, load_from_list=False, samples=None, is_test=False, unlabeled=False, choice=None):
            train_dataloader = DataLoader(train_data,
                                          shuffle=False,
                                          batch_size=1,
                                          pin_memory=False)

            total_predicts = []
            # total_score = []
            # total_every_score_list = []
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
                # every_score_list = []
                # for prob in probs[0]:
                #    if prob>0:
                #        every_score_list.append(prob)
                # total_every_score_list.append(every_score_list)
                predicts = predicts.detach().cpu().numpy()
                score = score.detach().cpu().numpy()

                # 转换为label_classes
                label_idx = list(batch["label_idx"][0].numpy())
                total_predicts += [label_idx.index(i) for i in predicts]

            logger.info("every_time:{}".format(time.time() - start1))
            y_true = []
            y_pred = []
            y_senetence = []
            for sample, predict in zip(vail_data, total_predicts):
                # if int(sample['label_id']) == current_label_id:
                if sample['label'] == current_label:
                    y_true.append(0)
                else:
                    y_true.append(1)
                y_pred.append(predict)
                if predict == 0:
                    y_senetence.append(sample['content'])

            trueY = np.array(y_true)
            testY = np.array(y_pred)
            logger.info("n {}".format(n))

            label = 0
            logger.info("f1_score {}".format(f1_score(trueY == label, testY == label, labels=True)))
            logger.info("recall {}".format(recall_score(trueY == label, testY == label, labels=True)))
            logger.info("precision {}".format(precision_score(trueY == label, testY == label, labels=True)))

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
            # two_choices[0] = '数学'
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
    # for input_label in input_labels:
    #     if input_label["current_label"] in labels_info:
    index = 0
    for value in input_labels["labels"]:
        if value in labels_info:
            current_label = value
            # result.append({"label":current_label, "rank":labels_info[current_label][0], "label_id":labels_info[current_label][1]})
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

    logger.info("label_detection_time: {}".format(time.time() - start))
    logger.info("label_detection_result {}".format(label_detection_result))
    logger.info("label_to_count {}".format(label_to_count))
    logger.info("len(sample_input_data) {}".format(len(sample_input_data)))
    logger.info("result {}".format(result))
    endtime = datetime.now()
    logger.info("RunTime: {}h-{}m-{}s".format(endtime.hour - starttime.hour,
                                              endtime.minute - starttime.minute,
                                              endtime.second - starttime.second))

    with open(result_path, 'w', encoding='utf-8') as f:
        for i in result:
            f.write(json.dumps(i, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    total_parser = argparse.ArgumentParser()
    total_parser.add_argument(
        "--model_path",
        default="/cognitive_comp/liuyibo/pretrained/pytorch",
        type=str,
        help="train task name")
    total_parser.add_argument(
        "--data_path",
        default="/cognitive_comp/zhubohan/GTS-Engine/examples/text_classification",
        type=str,
        help="train task name")

    total_parser.add_argument("--max_len", default=630, type=str)
    total_parser.add_argument("--num_labels", default=0, type=int)

    args = total_parser.parse_args()
    # args.data_path = '/cognitive_comp/liuyibo/zero_training_demo/'
    logger.info("args: {}".format(args))

    label_detection(model_path=os.path.join(args.model_path,
                                            'UnifiedMC_Bert-1.3B'),
                    data_path=args.data_path)

    # python inference.py --model_path /cognitive_comp/liuyibo/pretrained/pytorch/ --data_path /cognitive_comp/liuyibo/zero_training_demo/label_dete/demo/
