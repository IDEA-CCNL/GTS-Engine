

from termios import PARODD
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import Dataset, DataLoader
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

        self.train_data_path = os.path.join(args.data_dir, args.train_data)
        self.valid_data_path = os.path.join(args.data_dir, args.valid_data)
        self.test_data_path = os.path.join(args.data_dir, args.test_data)
        self.unlabeled_data_path = os.path.join(args.data_dir, args.unlabeled_data)


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
        return encoded

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

