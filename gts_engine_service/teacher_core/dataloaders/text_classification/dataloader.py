

from termios import PARODD
import time
import argparse
import itertools
import json
import copy
import os
from click import option
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning import Trainer, seed_everything, loggers
#from teacher_core.models.text_classification.bert_baseline import Bert
import sklearn
# from torchsnooper import snoop
from collections import OrderedDict
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TaskDataModel(pl.LightningDataModule):
    
    def __init__(self, args, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_workers = args.num_workers
        self.pretrained_model = args.pretrained_model
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize
        self.max_len = args.max_len
        self.task_name = args.task_name

        # self.cached_data_dir = os.path.join(args.data_dir, args.model_name)
        self.cached_data_dir = os.path.join(args.output_dir, "teacher_output/",args.task_name, "cached_data")
        if not os.path.exists(self.cached_data_dir):
            os.makedirs(self.cached_data_dir)

        self.cached_train_data_path = os.path.join(self.cached_data_dir, args.cached_train_data)
        self.cached_valid_data_path = os.path.join(self.cached_data_dir, args.cached_valid_data)
        self.cached_test_data_path = os.path.join(self.cached_data_dir, args.cached_test_data)
        self.cached_unlabeled_data_path = os.path.join(self.cached_data_dir, args.cached_unlabeled_data)

        self.train_data_path = os.path.join(args.data_dir, args.train_data)
        self.valid_data_path = os.path.join(args.data_dir, args.valid_data)
        self.test_data_path = os.path.join(args.data_dir, args.test_data)
        self.unlabeled_data_path = os.path.join(args.data_dir, args.unlabeled_data)
        if args.label2id_file is not None:
            self.label2id_file = os.path.join(args.data_dir, args.label2id_file)
        else:
            self.label2id_file = None

        # Whether to recreate dataset, useful when using a new pretrained model with different tokenizer, 
        # Default false, reuse cached data if exist
        self.recreate_dataset = args.recreate_dataset

        self.content_key = args.content_key
        self.label_key = args.label_key

        self.label_classes = self.get_label_classes(file_path=self.train_data_path,label_key=self.label_key)


    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_data = self.create_dataset(self.cached_train_data_path,
                                                 self.train_data_path)
            self.valid_data = self.create_dataset(self.cached_valid_data_path,
                                                 self.valid_data_path)
        if stage == 'test':
            self.test_data = self.create_dataset(self.cached_test_data_path,
                                                self.test_data_path,
                                                test=True)
            self.unlabeled_data = self.create_dataset(self.cached_unlabeled_data_path,
                                                self.unlabeled_data_path,
                                                test=True,
                                                unlabeled=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, 
        collate_fn=self.collate_fn, \
            batch_size=self.train_batchsize, num_workers=self.num_workers, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, 
        collate_fn=self.collate_fn, \
            batch_size=self.valid_batchsize, num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, 
        collate_fn=self.collate_fn, \
            batch_size=self.train_batchsize, num_workers=self.num_workers, pin_memory=False)
    
    def unlabeled_dataloader(self):
        return DataLoader(self.unlabeled_data, shuffle=False, 
        collate_fn=self.collate_fn, \
            batch_size=self.train_batchsize, num_workers=self.num_workers, pin_memory=False)

    def get_label_classes(self,file_path=None,label_key="label"):
        if self.label2id_file is not None:
            # print(self.label2id_file)
            with open(self.label2id_file, 'r', encoding='utf8') as f:
                label2id = json.load(f)
                # label_classes = list(label2id.keys())
                # 按照id键排序
                # label_classes = OrderedDict(sorted(label2id.items(), key=lambda i: i[1]['id']))
                # label_classes = list(label_classes.keys())
                # print(label_classes)
                # 用dict存储label_classes
                label_classes = {}
                for k, v in label2id.items():
                    label_classes[k] = v["id"]
        else:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.readlines()
                labels = []
                for line in tqdm(lines):
                    data = json.loads(line)
                    # text = data[content_key].strip("\n")
                    label = data[label_key] if label_key in data.keys() else 'unlabeled'  # 测试集中没有label标签，默认为0
                    # result.append((text, label))

                    if label not in labels:
                        labels.append(label)

                # 传入一个list，把每个标签对应一个数字
                label_model = sklearn.preprocessing.LabelEncoder()
                label_model.fit(labels)
                label_classes = {}
                for i,item in enumerate(list(label_model.classes_)):
                    label_classes[item] = i

        print("label_classes:",label_classes)
        return label_classes

    def load_data(self,file_path,content_key,label_key):
        with open(file_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            result = []
            labels = []
            for line in tqdm(lines):
                data = json.loads(line)
                text = data[content_key].strip("\n")
                if len(text) > self.max_len:
                    text = text[:int(self.max_len/2)] + text[int(self.max_len/2):]
                label = data[label_key] if label_key in data.keys() else 'unlabeled'  # 测试集中没有label标签，默认为0
                data_id = data["id"] if "id" in data.keys() else '0' 
                result.append((text, label, data_id))

        result2 = []
        class_cnt = {}
        for res in result:
            if res[1] == "unlabeled":
                class_index = 0
            else:
                class_index = self.label_classes[res[1]]
            # sentence, label, idx
            result2.append([res[0],class_index, res[2]])
            if class_index not in class_cnt.keys():
                class_cnt[class_index] = 1
            else:
                class_cnt[class_index] += 1

        print(class_cnt)
        return result2

    def encode(self, sentence):
        # Do not return_tensors here, otherwise rnn.pad_sequence in collate_fn will raise error
        # encoded = self.tokenizer(sentence, truncation=True, max_length=512)
        # encoded = self.tokenizer(sentence, truncation=True, padding="max_length", max_length=self.max_len)
        # padding=longest让每个batch不一样长
        encoded = self.tokenizer(sentence, truncation=True, padding="longest", max_length=self.max_len)
        
        encoded['sentence'] = sentence
        encoded['input_ids'] = torch.LongTensor(encoded['input_ids'])
        encoded['attention_mask'] = torch.LongTensor(encoded['attention_mask'])
        encoded['token_type_ids'] = torch.LongTensor(encoded['token_type_ids'])

        #  Customize your example here if needed
        #  input_ids = encoded['input_ids']
        #  attention_mask = encoded['attention_mask']
        #  Models like roberta don't have token_type_ids
        #  if 'token_type_ids' not in encoded:
        #      encoded['token_type_ids'] = [[0] * len(x) for x in input_ids]
        #  example = {
        #      'sentence': sentence,
        #      'input_ids': torch.LongTensor(input_ids),
        #      'attention_mask': torch.LongTensor(attention_mask),
        #      'token_type_ids': torch.LongTensor(encoded['token_type_ids']),
        #  }

        

        return encoded

    def create_dataset(self, cached_data_path='', data_path=None, test=False, unlabeled=False,load_from_list=False, sentence_list=[]):
        if  os.path.exists(cached_data_path) and not self.recreate_dataset:
            print(f'Loading cached dataset from {cached_data_path}...')
            data = torch.load(cached_data_path)
            #  Filter data if you don't need all of them
            #  data = list(filter(lambda x: len(self.acronym2lf[x['acronym']]) < 15 and (x['acronym'] in self.ori_diction or random.random() < 0.2), data))
            output = f'Load {len(data)} instances from {cached_data_path}.'
        elif load_from_list:
            data = []
            for sentence in tqdm(sentence_list):
                encoded = self.encode(sentence)
                encoded["id"] = 0
                data.append(encoded)

            data = TaskDataset(data)
            output = f'Load {len(data)} instances from sentence_list.'
            print('Last example:', encoded)
        else:
            print(f'Preprocess {data_path} for {self.task_name}...')
            # dataset = json.load(open(data_path, 'r'))
            dataset = self.load_data(file_path=data_path,content_key=self.content_key,label_key=self.label_key)
            data = []

            for example in tqdm(dataset):
                encoded = self.encode(example[0])
                encoded["id"] = example[2]
                if not unlabeled:
                    label = int(example[1])
                    encoded['labels'] = label
                data.append(encoded)
            print(data[0])
            output = f'Load {len(data)} instances from {data_path}.'
            data = TaskDataset(data)
            torch.save(data, cached_data_path)
            print('Last example:', encoded)

        print(output)
        return data

    def collate_fn(self, batch):
        '''
        Aggregate a batch data.
        batch = [ins1_dict, ins2_dict, ..., insN_dict]
        batch_data = {'sentence':[ins1_sentence, ins2_sentence...], 'input_ids':[ins1_input_ids, ins2_input_ids...], ...}
        '''
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]
        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        token_type_ids = batch_data['token_type_ids']
        labels = None
        if 'labels' in batch_data:
            labels = torch.LongTensor(batch_data['labels'])

        # Before pad input_ids = [tensor<seq1_len>, tensor<seq2_len>, ...]
        # After pad input_ids = tensor<batch_size, max_seq_len>
        input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                              batch_first=True,
                                              padding_value=self.tokenizer.pad_token_id)
        attention_mask = nn.utils.rnn.pad_sequence(attention_mask,
                                                   batch_first=True,
                                                   padding_value=0)
        token_type_ids = nn.utils.rnn.pad_sequence(token_type_ids,
                                                   batch_first=True,
                                                   padding_value=0)

        batch_data = {
            "id": batch_data["id"],
            'sentence': batch_data['sentence'],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels,
        }

        # print(batch)

        return batch_data


class TaskDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    total_parser = argparse.ArgumentParser()

    # * Args for data preprocessing
    total_parser = TaskDataModel.add_data_specific_args(total_parser)
    
    # * Args for training
    #  total_parser = Trainer.add_argparse_args(total_parser)

    # * Args for model specific
    total_parser = Bert.add_model_specific_args(total_parser)

    args = total_parser.parse_args()


    # * Here, we test the data preprocessing
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model,
                                              use_fast=True)

    task2_data = TaskDataModel(args, tokenizer)

    task2_data.setup('fit')
    task2_data.setup('test')

    val_dataloader = task2_data.val_dataloader()

    batch = next(iter(val_dataloader))

    print(batch)

