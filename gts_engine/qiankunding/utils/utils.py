
import json
import numpy as np

from gts_common.logs_utils import Logger

logger = Logger().get_log()


def truncate_sequences(maxlen, indices, *sequences):
    """
    截断直至所有的sequences总长度不超过maxlen
    参数:
        maxlen:
            所有sequence长度之和的最大值 
        indices:
            truncate时删除单词在sequence中的位置
        sequences:
            一条或多条文本
    """
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            # 对sequence中最长的那个进行truncate
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


def load_json_by_line(file):
    data = []
    with open(file, "r", encoding="utf8") as f:
        reader = f.readlines()
        for line in reader:
            
            sample = json.loads(line.strip())
            data.append(sample)
    return data


def load_json(json_file):
    with open(json_file, "r", encoding="utf8") as fin:
        data = json.load(fin)
    logger.info("load {} from {}".format(len(data), json_file))
    return data

def write2json(data_list, data_path, data_name):
    with open(data_path, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(data_list, ensure_ascii=False, indent=2))
        logger.info("{}({}) saved into {}".format(data_name, len(data_list), data_path))

def write2json_by_line(data_list, data_path, data_name=""):
    with open(data_path, "w", encoding="utf-8") as fout:
        for result in data_list:
            
            result = json.dumps(result,ensure_ascii=False)
            fout.write(result+"\n")
        logger.info("{}({}) saved into {}".format(data_name, len(data_list), data_path))

def json2list(data_path, use_key):
    data_list = []
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            sample_dict = {}
            item = json.loads(line)
            for key in use_key:
                sample_dict[key] = item[key]
            data_list.append(sample_dict)
    return data_list


def list2json(data_list, data_path, use_key):
    with open(data_path, "w", encoding="utf-8") as fout:
        for idx, result in enumerate(data_list):
            
            sample_dict = {}
            result["id"] = idx
            for key in use_key:
                sample_dict[key] = result[key]
            result = json.dumps(sample_dict,ensure_ascii=False)
            fout.write(result+"\n")

    

