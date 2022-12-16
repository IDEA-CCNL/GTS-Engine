
import json
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
import sklearn
from torch.utils.data import Dataset, DataLoader

from gts_common.logs_utils import Logger

logger = Logger().get_log()
logger.propagate = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def random_masking(token_ids, maks_rate, mask_start_idx, max_length, mask_id, tokenizer):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []

    for i, (r, t) in enumerate(zip(rands, token_ids)):
        if i < mask_start_idx:
            source.append(t)
            target.append(-100)
            continue
        if r < maks_rate * 0.8:
            source.append(mask_id)
            target.append(t)
        elif r < maks_rate * 0.9:
            source.append(t)
            target.append(t)
        elif r < maks_rate:
            source.append(np.random.choice(len(tokenizer) - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(-100)
    while len(source) < max_length:
        source.append(0)
        target.append(-100)
    return source[:max_length], target[:max_length]



def get_att_mask(attention_mask,label_idx,question_len):
    max_length=len(attention_mask)
    attention_mask=np.array(attention_mask)
    attention_mask=np.tile(attention_mask[None, :],(max_length,1))

    zeros=np.zeros(shape=(label_idx[-1]-question_len,label_idx[-1]-question_len))
    attention_mask[question_len:label_idx[-1],question_len:label_idx[-1]]=zeros

    for i in range(len(label_idx)-1):
        label_token_length=label_idx[i+1]-label_idx[i]
        if label_token_length<=0:
            logger.info("label_idx {}".format(label_idx))
            logger.info("question_len {}".format(question_len))
            continue
        ones=np.ones(shape=(label_token_length,label_token_length))
        attention_mask[label_idx[i]:label_idx[i+1],label_idx[i]:label_idx[i+1]]=ones

    return attention_mask


def get_position_ids(label_idx,max_length,question_len):
    question_position_ids=np.arange(question_len)
    label_position_ids=np.arange(question_len,label_idx[-1])
    for i in range(len(label_idx)-1):
        label_position_ids[label_idx[i]-question_len:label_idx[i+1]-question_len]=np.arange(question_len,question_len+label_idx[i+1]-label_idx[i])
    max_len_label=max(label_position_ids)
    text_position_ids=np.arange(max_len_label+1,max_length+max_len_label+1-label_idx[-1])
    position_ids=list(question_position_ids)+list(label_position_ids)+list(text_position_ids)
    if max_length<=512:
        return position_ids[:max_length]
    else:
        for i in range(512,max_length):
            if position_ids[i]>511:
                position_ids[i]=511
        return position_ids[:max_length]


def get_token_type(sep_idx,max_length):
    token_type_ids=np.zeros(shape=(max_length,))
    for i in range(len(sep_idx)-1):
        if i%2==0:
            ty=np.ones(shape=(sep_idx[i+1]-sep_idx[i],))
        else:
            ty=np.zeros(shape=(sep_idx[i+1]-sep_idx[i],))
        token_type_ids[sep_idx[i]:sep_idx[i+1]]=ty
    
    return token_type_ids

class TaskDatasetUnifiedMCForNLI(Dataset):
    def __init__(self, data_path=None, args=None,used_mask=True, tokenizer=None, load_from_list=False, samples=None, is_test=False, unlabeled=False):
        super().__init__()
        # added_token=['[unused'+str(i+1)+']' for i in range(99)]
        # self.tokenizer = BertTokenizer.from_pretrained(
        #     args.pretrained_model_path,additional_special_tokens=added_token)

        self.tokenizer = tokenizer
        self.yes_token = self.tokenizer.encode("是")[1]
        self.no_token = self.tokenizer.encode("非")[1]


        self.max_length = args.max_len
        self.num_labels = args.num_labels
        self.used_mask = used_mask
        self.args = args
        self.args.use_label_attention_mask='True'
        self.args.use_align_position='True'

        self.load_from_list = load_from_list
        self.samples = samples
        self.is_test = is_test
        self.unlabeled = unlabeled

        self.data = self.load_data(data_path, args, load_from_list, samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode(self.data[index],self.used_mask, self.is_test, self.unlabeled)

    def ori2mrc(self,item):
        new_item = {}
        new_item["id"] = item["id"]
        new_item["texta"] = item["sentence1"]
        new_item["textb"] = item["sentence2"]
        textb = item["sentence2"]
        new_item["question"] = "根据这段话"
        new_item["choice"] = [f"可以推断出：{textb}",f"不能推断出：{textb}",f"很难推断出：{textb}"]

        label2id = {"entailment":0, "contradiction":1, "neutral":2}
        new_item["label"] = label2id[item["label"]]
        new_item["answer"] = new_item["choice"][new_item["label"]]

        return new_item


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
                    new_item = self.ori2mrc(item)
                    samples.append(new_item)

        return samples

    def encode(self, item, used_mask, is_test, unlabeled):
        # item['question']='[CLS]'
        # 如果choice太长的处理
        while len(self.tokenizer.encode('[MASK]'.join(item['choice']))) > self.max_length-32:
            item['choice'] = [c[:int(len(c)/2)] for c in item['choice']]

        texta =  '[MASK]' + '[MASK]'.join(item['choice'])+ '[SEP]'+item['question'] + '[SEP]' +item['texta']
        encode_dict = self.tokenizer.encode_plus(texta,
                                            max_length=self.max_length,
                                            # padding='max_length',
                                            padding="longest",
                                            truncation=True
                                            )
        
        encode_sent = encode_dict['input_ids']
        token_type_ids=encode_dict['token_type_ids']
        attention_mask=encode_dict['attention_mask']
        
        # question_len=len(self.tokenizer.encode(item['question']))
        question_len=1
        label_idx=[question_len]
        for choice in item['choice']:
            cur_mask_idx=label_idx[-1]+len(self.tokenizer.encode(choice,add_special_tokens=False))+1
            if cur_mask_idx < self.max_length:
                label_idx.append(cur_mask_idx)
    

        # token_type_ids=[0]*question_len+[1]*(label_idx[-1]-label_idx[0]+1)+[0]*self.max_length
        # token_type_ids=token_type_ids[:self.max_length]
        encoded_len = len(encode_dict["input_ids"])
        zero_len = len(encode_dict["input_ids"]) - question_len - ((label_idx[-1]-label_idx[0]+1))
        token_type_ids=[0]*question_len+[1]*(label_idx[-1]-label_idx[0]+1)+[0]*zero_len

        if self.args.use_label_attention_mask=='True':

            attention_mask=get_att_mask(attention_mask,label_idx,question_len)

        if self.args.use_align_position=='True':
            try:
                position_ids=get_position_ids(label_idx,encoded_len,question_len)
            except:
                logger.info(item)
        else:
            position_ids=np.arange(self.max_length)
        
        clslabels_mask = np.zeros(shape=(len(encode_sent),))
        clslabels_mask[label_idx[:-1]]=10000
        clslabels_mask=clslabels_mask-10000

        mlmlabels_mask=np.zeros(shape=(len(encode_sent),))
        mlmlabels_mask[label_idx[0]]=1

        # used_mask=False
        if used_mask:
            mask_rate = 0.1*np.random.choice(4,p=[0.3,0.3,0.25,0.15])
            source, target = random_masking(token_ids=encode_sent, maks_rate=mask_rate,
                                            mask_start_idx=label_idx[-1], max_length=encoded_len, 
                                            mask_id=self.tokenizer.mask_token_id,
                                            tokenizer=self.tokenizer)
        else:
            source, target = encode_sent[:], encode_sent[:]
        
        source=np.array(source)
        target=np.array(target)
        source[label_idx[:-1]]=self.tokenizer.mask_token_id
        target[label_idx[:-1]]=self.no_token
        if unlabeled:
            # rand_idx = random.sample(label_idx,1)[0]
            rand_idx = label_idx[0]
            target[rand_idx]=self.yes_token
            clslabels=label_idx[0]
        else:
            target[label_idx[item['label']]]=self.yes_token
            clslabels = label_idx[item['label']]
        # target[label_idx[:-1]]=-100
        # target[label_idx[item['label']]]=-100


        encoded = {
            "id": item["id"],
            "texta":item["texta"],
            "textb":item["textb"],
            "question":item["question"],
            "choice":item["choice"],
            "unlabeled_set":item["unlabeled_set"] if "unlabeled_set" in item.keys() else "0",
            "input_ids": torch.tensor(source).long(),
            "token_type_ids": torch.tensor(token_type_ids).long(),
            "attention_mask": torch.tensor(attention_mask).float(),
            "position_ids":torch.tensor(position_ids).long(),
            "mlmlabels": torch.tensor(target).long(),
            "clslabels": torch.tensor(clslabels).long(),
            "clslabels_mask": torch.tensor(clslabels_mask).float(),
            "mlmlabels_mask": torch.tensor(mlmlabels_mask).float(),
            "label_idx": torch.tensor(label_idx).long(),
            "use_mask": used_mask,
        }

        return encoded


class TaskDataModelUnifiedMCForNLI(pl.LightningDataModule):
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
        parser.add_argument('--knn_datastore_data', default='train.json', type=str)
        parser.add_argument('--max_len', default=128, type=int)

        parser.add_argument('--texta_name', default='text', type=str)
        parser.add_argument('--textb_name', default='sentence2', type=str)
        parser.add_argument('--content_key', default="content",help="content key in json file")
        parser.add_argument('--label_key', default="label",help="label key in json file")

        return parent_args

    def __init__(self, args, tokenizer):
        super().__init__()
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize
        self.test_batchsize = 5
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer


        self.label_classes = self.get_label_classes(file_path=os.path.join(args.data_dir, args.train_data))
        args.num_labels = len(self.label_classes)

        self.train_data = TaskDatasetUnifiedMCForNLI(os.path.join(
            args.data_dir, args.train_data), args, used_mask=True, tokenizer=tokenizer, is_test=False, unlabeled=False)
        self.valid_data = TaskDatasetUnifiedMCForNLI(os.path.join(
            args.data_dir, args.valid_data), args, used_mask=False, tokenizer=tokenizer, is_test=True, unlabeled=False)
        self.test_data = TaskDatasetUnifiedMCForNLI(os.path.join(
            args.data_dir, args.test_data), args, used_mask=False, tokenizer=tokenizer, is_test=True, unlabeled=False)
        logger.info("len(valid_data): {}".format(len(self.valid_data)))
       
    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, collate_fn=self.collate_fn, batch_size=self.train_batchsize, pin_memory=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, collate_fn=self.collate_fn, batch_size=self.valid_batchsize, pin_memory=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, collate_fn=self.collate_fn, batch_size=self.test_batchsize, pin_memory=False, num_workers=self.num_workers)

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
        token_type_ids = batch_data["token_type_ids"]
        position_ids = batch_data["position_ids"]
        mlmlabels = batch_data["mlmlabels"]
        mlmlabels_mask = batch_data["mlmlabels_mask"]
        clslabels_mask = batch_data["clslabels_mask"]

        # Before pad input_ids = [tensor<seq1_len>, tensor<seq2_len>, ...]
        # After pad input_ids = tensor<batch_size, max_seq_len>
        input_ids = nn.utils.rnn.pad_sequence(input_ids,
                                              batch_first=True,
                                              padding_value=self.tokenizer.pad_token_id)
        max_len = input_ids.size(1)
        attention_mask_ = []
        for item in  attention_mask:
            item_len=item.size(0)
            new_item = torch.nn.functional.pad(input=item, 
            pad=(
                0,max_len-item_len, # 在右边填充
                0,max_len-item_len, # 在下边填充
            ),
            mode="constant",
            value=0
            )
            attention_mask_.append(new_item)

        new_attention_mask = torch.stack(attention_mask_,dim=0)
            

        token_type_ids = nn.utils.rnn.pad_sequence(token_type_ids,
                                                   batch_first=True,
                                                   padding_value=0)
        position_ids = nn.utils.rnn.pad_sequence(position_ids,
                                                   batch_first=True,
                                                   padding_value=0)
                                    
        mlmlabels = nn.utils.rnn.pad_sequence(mlmlabels,
                                                batch_first=True,
                                                padding_value=0)
        mlmlabels_mask = nn.utils.rnn.pad_sequence(mlmlabels_mask,
                                                batch_first=True,
                                                padding_value=0)
        clslabels_mask = nn.utils.rnn.pad_sequence(clslabels_mask,
                                                batch_first=True,
                                                padding_value=-10000)

        clslabels = torch.stack(batch_data["clslabels"],dim=0)

        label_idx = torch.stack(batch_data["label_idx"],dim=0)


        batch_data = {
            "id":batch_data["id"],
            "texta":batch_data["texta"],
            "textb":batch_data["textb"],
            "question":batch_data["question"],
            "choice":batch_data["choice"],
            "unlabeled_set": batch_data["unlabeled_set"],
            "input_ids": input_ids,
            "attention_mask": new_attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "mlmlabels": mlmlabels,
            "clslabels": clslabels,
            "clslabels_mask": clslabels_mask,
            "mlmlabels_mask": mlmlabels_mask,
            "label_idx": label_idx,
            "use_mask": batch_data["use_mask"],
        }

        # logger.info(batch)

        return batch_data

    def get_label_classes(self,file_path=None,label_key="label"):
    
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

        logger.info("label_classes: {}".format(label_classes))
        return label_classes
