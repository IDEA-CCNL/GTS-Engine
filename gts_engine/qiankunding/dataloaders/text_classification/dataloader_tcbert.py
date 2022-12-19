import json
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from gts_common.logs_utils import Logger

logger = Logger().get_log()

os.environ["TOKENIZERS_PARALLELISM"] = "false"



class TaskDatasetTCBert(Dataset):
    def __init__(self, data_path=None, args=None, tokenizer=None, load_from_list=False, samples=None,  unlabeled=False, label_classes=None):
        super().__init__()

        self.tokenizer = tokenizer

        self.max_length = args.max_len
        self.args = args

        self.load_from_list = load_from_list
        self.samples = samples
        self.unlabeled = unlabeled

        self.data = self.load_data(data_path, args, load_from_list, samples)
        self.label_classes = label_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode(self.data[index], self.unlabeled)

    def load_data(self, data_path, args=None, load_from_list=False, sentences=None):
        samples = []
        if load_from_list:
            for line in tqdm(sentences):
                samples.append(line)
        else:
            with open(data_path, 'r', encoding='utf8') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    item = json.loads(line)
                    samples.append(item)
        return samples

    def encode(self, item, unlabeled):

        texta = '这一句描述{}的内容如下：'.format('[MASK][MASK]')  + item['content']

        encode_dict = self.tokenizer.encode_plus(texta,
                                            max_length=self.max_length,
                                            # padding='max_length',
                                            padding="longest",
                                            truncation=True
                                            )
        
        input_ids = encode_dict['input_ids']
        token_type_ids = encode_dict['token_type_ids']
        attention_mask = encode_dict['attention_mask']


        if not unlabeled:
            labels = self.label_classes[item['label']]

            encoded = {
                "sentence":item["content"],
                "input_ids": torch.tensor(input_ids).long(),
                "token_type_ids": torch.tensor(token_type_ids).long(),
                "attention_mask": torch.tensor(attention_mask).float(),
                "labels": torch.tensor(labels).long()
            }

        else:
            encoded = {
                "sentence":item["content"],
                "input_ids": torch.tensor(input_ids).long(),
                "token_type_ids": torch.tensor(token_type_ids).long(),
                "attention_mask": torch.tensor(attention_mask).float(),
            }
        return encoded


class TaskDataModelTCBert(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('TASK NAME DataModel')
        parser.add_argument('--data_dir', default='./data', type=str)
        parser.add_argument('--train_data', default='train.json', type=str)
        parser.add_argument('--valid_data', default='dev.json', type=str)
        parser.add_argument('--test_data', default='test.json', type=str)
        parser.add_argument('--train_batchsize', default=16, type=int)
        parser.add_argument('--valid_batchsize', default=32, type=int)
        parser.add_argument('--unlabeled_data', default='unlabeled.json', type=str)
        parser.add_argument('--max_len', default=128, type=int)

        return parent_args

    def __init__(self, args, tokenizer):
        super().__init__()
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize
        # self.test_batchsize = 5
        self.test_batchsize = args.test_batchsize
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer

        self.choice, self.label_classes = self.get_label_classes(file_path=os.path.join(args.data_dir, args.label_data))
        args.num_labels = len(self.choice)

        self.train_data = TaskDatasetTCBert(os.path.join(
            args.data_dir, args.train_data), args,  tokenizer=tokenizer, unlabeled=False, label_classes=self.label_classes)
        self.valid_data = TaskDatasetTCBert(os.path.join(
            args.data_dir, args.valid_data), args, tokenizer=tokenizer, unlabeled=False, label_classes=self.label_classes)     
        self.unlabeled_data = TaskDatasetTCBert(os.path.join(
            args.data_dir, args.unlabeled_data), args, tokenizer=tokenizer, unlabeled=True, label_classes=self.label_classes)
        logger.info("unlabeled_data_len: {}".format(len(self.unlabeled_data)))

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, collate_fn=tcbert_collate_fn, batch_size=self.train_batchsize, pin_memory=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, collate_fn=tcbert_collate_fn, batch_size=self.valid_batchsize, pin_memory=False, num_workers=self.num_workers)

    def unlabeled_dataloader(self):
        return DataLoader(self.unlabeled_data, shuffle=False, collate_fn=tcbert_collate_fn, batch_size=self.test_batchsize, pin_memory=False, num_workers=self.num_workers)

    def get_label_classes(self,file_path=None):
        
        line = json.load(open(file_path, 'r', encoding='utf8'))
        choice = line['labels']

        label_classes = {}
        for i, item in enumerate(choice):
            label_classes[item] = i

        logger.info("label_classes: {}".format(label_classes))
        logger.info("choice: {}".format(choice))
        return choice, label_classes

def tcbert_collate_fn(batch):
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
    token_type_ids = batch_data["token_type_ids"]
    labels = None
    if 'labels' in batch_data:
        labels = torch.LongTensor(batch_data['labels'])
    
    input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                            batch_first=True,
                                            padding_value=0)
        

    token_type_ids = nn.utils.rnn.pad_sequence(token_type_ids,
                                                batch_first=True,
                                                padding_value=0)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask,
                                                batch_first=True,
                                                padding_value=0)

    batch_data = {
        "sentence":batch_data["sentence"],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        'labels': labels,
    }

    return batch_data