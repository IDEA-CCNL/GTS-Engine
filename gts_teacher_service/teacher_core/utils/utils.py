
import json
import numpy as np


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
            # print(line)
            sample = json.loads(line.strip())
            data.append(sample)
    return data


def load_json(json_file):
    with open(json_file, "r", encoding="utf8") as fin:
        data = json.load(fin)
    print("load {} from {}".format(len(data), json_file))
    return data

def write2json(data_list, data_path, data_name):
    with open(data_path, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(data_list, ensure_ascii=False, indent=2))
        print("{}({}) saved into {}".format(data_name, len(data_list), data_path))

def write2json_by_line(data_list, data_path, data_name=""):
    with open(data_path, "w", encoding="utf-8") as fout:
        for result in data_list:
            # print(result["id"])
            result = json.dumps(result,ensure_ascii=False)
            fout.write(result+"\n")
        print("{}({}) saved into {}".format(data_name, len(data_list), data_path))


    

