import json
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from gts_common.logs_utils import Logger
logger = Logger().get_log()

class TaskDatasetKGT5(torch.utils.data.Dataset):
    def __init__(self, data_path=None, args=None, tokenizer=None, load_from_list=False, samples=None):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = args.max_len
        self.args = args
        self.load_from_list = load_from_list
        self.samples = samples
        self.data = self.load_data(data_path, args, load_from_list, samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode(self.data[index])

    def load_data(self, data_path, args=None, load_from_list=False, sentences=None):
        samples = []

        if load_from_list:
            for line in tqdm(sentences):
                samples.append(line)
        else:
            with open(data_path, 'r', encoding='utf8') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    samples.append(json.loads(line))
        return samples

    def encode(self, item):
        
        text = f"""这段文本的关键词是?""" + f"""【{item["content"]}】"""
        label = item["label"]

        encode_dict = self.tokenizer(text, max_length=self.max_length, padding='longest',truncation=True)
        decode_dict = self.tokenizer(label, max_length=self.max_length // 2, padding='longest',truncation=True)

        encoded = {
            "id":item["id"],
            "sentence":text,
            "input_ids": torch.tensor(encode_dict['input_ids']).long(),
            "attention_mask": torch.tensor(encode_dict['attention_mask']).long(),
            "labels": torch.tensor(decode_dict['input_ids']).long(),
        }
        return encoded


class TaskDataModelKGT5(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('TASK NAME DataModel')
        parser.add_argument('--data_dir', default='./data', type=str)
        parser.add_argument('--train_data', default='train.json', type=str)
        parser.add_argument('--valid_data', default='dev.json', type=str)
        parser.add_argument('--test_data', default='test.json', type=str)
        parser.add_argument('--train_batchsize', default=16, type=int)
        parser.add_argument('--valid_batchsize', default=32, type=int)
        parser.add_argument('--max_len', default=128, type=int)

        return parent_args

    def __init__(self, args, tokenizer):
        super().__init__()
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize
        self.num_workers = args.num_workers
        self.test_batchsize = args.test_batchsize
        self.tokenizer = tokenizer

        self.train_data = TaskDatasetKGT5(os.path.join(
            args.data_dir, args.train_data), args, tokenizer=tokenizer)
        self.valid_data = TaskDatasetKGT5(os.path.join(
            args.data_dir, args.valid_data), args, tokenizer=tokenizer)
        self.test_data = TaskDatasetKGT5(os.path.join(
            args.data_dir, args.test_data), args, tokenizer=tokenizer)
        

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, collate_fn=kg_collate_fn, batch_size=self.train_batchsize, pin_memory=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, collate_fn=kg_collate_fn, batch_size=self.valid_batchsize, pin_memory=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, collate_fn=kg_collate_fn, batch_size=self.test_batchsize, pin_memory=False, num_workers=self.num_workers)
    

def kg_collate_fn(batch):
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
    labels = batch_data["labels"]


    input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                            batch_first=True,
                                            padding_value=0)

    new_attention_mask = nn.utils.rnn.pad_sequence(attention_mask,
                                                batch_first=True,
                                                padding_value=0)

    labels = nn.utils.rnn.pad_sequence(labels,
                                        batch_first=True,
                                        padding_value=-100)

    kpg_labels = labels.clone()
    kpg_labels[labels < 0] = 0

    batch_data = {
        "id":batch_data["id"],
        "sentence":batch_data["sentence"],
        "input_ids": input_ids,
        "attention_mask": new_attention_mask,
        "labels": labels,
        "kpg_labels": kpg_labels,
    }

    return batch_data