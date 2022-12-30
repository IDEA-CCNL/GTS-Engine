from typing import List, Optional, Tuple
import re
from LAC import LAC
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np




def cut_sent(para):
    """分字并处理标点符号"""
    para = re.sub('([，。！？,\.\?!])([^”’])', r"\1\n\2", para)  # 单字符断句符 
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

lac_seg = LAC(mode='lac')

# TODO 未改动，需要优化
def segment_text(text: str, tokenizer: Optional[PreTrainedTokenizer] = None)-> List[str]: 
    """
    输入一句话，返回一句经过处理的分词列表: 为了支持中文全称mask，将被分开的词，将上特殊标记("#")，使得后续处理模块，能够知道哪些字是属于同一个词的。
    """
    #text = "".join(cut_sent(text)) # 处理标点符号
    seq_cws = lac_seg.run(text)[0]
    seq_cws_dict = {x: 1 for x in seq_cws}  # 分词后的词加入到词典dict
    
    segment: List[str] = []
    i = 0
    while i < len(text):  # 从句子的第一个字开始处理，知道处理完整个句子
        '''if len(re.findall('[\u4E00-\u9FA5]', text[i])) == 0:  # 如果找不到中文的，原文加进去即不用特殊处理。
            new_text.append(text[i])
            i += 1
            continue'''

        # 由于LAC对于像2018或者abc这种非中文字符会直接切分成单个字符，所以这里加入连续非中文字符的识别，然后使用bert的分词器进行重新切分
        non_chinese = []
        while i < len(text) and len(re.findall('[\u4E00-\u9FA5]', text[i])) == 0:
            non_chinese.append(text[i])
            i += 1
        if non_chinese:
            tokens = tokenizer.tokenize(
                ''.join(non_chinese)) if tokenizer else non_chinese
            segment.extend(tokens)
        if i >= len(text):
            break

        has_add = False
        for length in range(3, 0, -1):
            if i + length > len(text):
                continue
            if ''.join(text[i:i + length]) in seq_cws_dict:
                segment.append(text[i])
                for l in range(1, length):
                    segment.append('##' + text[i + l])
                i += length
                has_add = True
                break
        if not has_add:
            segment.append(text[i])
            i += 1
    return segment

# TODO 未改动，需要优化
def text_2_training_instance(text: str, max_length: int, tokenizer: PreTrainedTokenizer, index: Optional[int]=None) -> Tuple[List[str], Optional[bool]]: # type: ignore
    # char_list = segment_text(text, tokenizer) .   # 用来分句的
    char_list = [segment_text(sen, tokenizer) for sen in cut_sent(text)]
    max_num_tokens = max_length - 3
    target_seq_length = max_num_tokens
    if index is None:
        index = np.random.randint(1, 10000)
    rng = np.random.RandomState(seed=((1234 + index) % 2 ** 32))  # type: ignore
    current_chunk = []  # 当前处理的文本段，包含多个句子
    current_length = 0
    i = 0

    tokens = []
    is_random_next = None

    while i < len(char_list):  # 从文档的第一个位置开始，按个往下看
        # segment是列表，代表的是按字分开的一个完整句子，
        # segment=['我', '是', '一', '爷', '##们', '，', '我', '想', '我', '会', '给', '我',
        # '媳', '##妇', '最', '##好', '的', '##幸', '##福', '。']
        segment = char_list[i]

        current_chunk.append(segment)  # 将一个独立的句子加入到当前的文本块中
        current_length += len(segment)  # 累计到为止位置接触到句子的总长度
        if i == len(char_list) - 1 or current_length >= target_seq_length:
            # 如果累计的序列长度达到了目标的长度，或当前走到了文档结尾==>构造并添加到“A[SEP]B“中的A和B中；
            if current_chunk:  # 如果当前块不为空
                a_end = 1
                # 当前块，如果包含超过两个句子，取当前块的一部分作为“A[SEP]B“中的A部分
                if len(current_chunk) > 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)
                elif len(current_chunk) == 2:
                    a_end = 1

                # 将当前文本段中选取出来的前半部分，赋值给A即tokens_a
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                # 构造“A[SEP]B“中的B部分(有一部分是正常的当前文档中的后半部;在原BERT的实现中一部分是随机的从另一个文档中选取的，）
                tokens_b = []
                for j in range(a_end, len(current_chunk)):
                    tokens_b.extend(current_chunk[j])

                # 有百分之50%的概率交换一下tokens_a和tokens_b的位置
                # print("tokens_a length1:",len(tokens_a))
                # print("tokens_b length1:",len(tokens_b)) # len(tokens_b) = 0

                if len(tokens_a) == 0 or len(tokens_b) == 0:
                    tokens = tokens_a[:max_num_tokens]
                    tokens.append('[SEP]')
                    return tokens, None

                if rng.random() < 0.5:  # 交换一下tokens_a和tokens_b
                    is_random_next = True
                    temp = tokens_a
                    tokens_a = tokens_b
                    tokens_b = temp
                else:
                    is_random_next = False

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                # 把tokens_a & tokens_b加入到按照bert的风格，即以[CLS]tokens_a[SEP]tokens_b[SEP]的形式，结合到一起，
                # 作为最终的tokens; 也带上segment_ids，前面部分segment_ids的值是0，后面部分的值是1.
                tokens = []
                segment_ids = []
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                segment_ids.append(1)

            return tokens, is_random_next

        i += 1  # 接着文档中的内容往后看
    
    
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()
    
    
    