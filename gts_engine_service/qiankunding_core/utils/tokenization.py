

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open
from typing import List, Union, Dict, Set, Tuple, Optional
from .utils import truncate_sequences
from transformers import AutoTokenizer, BertTokenizer, T5Tokenizer
from transformers.models.bart.tokenization_bart import BartTokenizer

logger = logging.getLogger(__name__)

def get_train_tokenizer(args):
   
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    tokenizer = get_unused_tokenizer(args.pretrained_model)


    return tokenizer


def get_unused_tokenizer(pretrained_model_path):
    """添加特殊中文字符和未使用的token【unused1】"""
    if 'ernie' not in pretrained_model_path:
        added_token = ['[unused'+str(i+1)+']' for i in range(200)]
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path,
                                        additional_special_tokens=added_token)
    else:
        added_token = ['[unused'+str(i+1)+']' for i in range(200)]
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path,
                            additional_special_tokens=added_token)
    return tokenizer


def load_vocab(vocab_file):
    """加载vocab文件为dict"""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index

    return vocab


def whitespace_tokenize(text):
    """去除文本中的空白符，并按照空格分词"""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class Tokenizer(object):
    def __init__(
        self, 
        vocab_file, 
        do_lower_case=True, 
        do_basic_tokenize=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]"):
        """
        类似于BertTokenizer的Tokenizer
        参数:
            vocab_file:
                词典文件
            do_lower_case:
                是否转换成小写
            do_basic_tokenize:
                分词前，是否进行基础的分词
            unk_token:
                未知词标记
            sep_token:
                句子切分标记，当只有一句话作为输入时，此标记知识作为结束符；当有两句话作为输入时，此标记作为分隔符、最后一句话的结束符
            pad_token:
                padding填充标记
            cls_token:
                分类标记，位于整个序列的第一个
            mask_token:
                mask标记
        """
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                  never_split=(unk_token, sep_token, pad_token, cls_token, mask_token))
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

    def tokenize(self, text: str) -> List[str]:
        """先进行basic tokenize，再进行word piece tokenize，最后添加起始和末尾token [CLS] [SEP]"""
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                    for sub_token in self.wordpiece_tokenizer.tokenize(token):
                        split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        if self.cls_token is not None:
            split_tokens.insert(0, self.cls_token)
        if self.sep_token is not None:
            split_tokens.append(self.sep_token)

        return split_tokens

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def encode(self, 
               first_text: Union[str, List[str]],
               second_text: Optional[Union[str, List[str]]] = None,
               max_len: int = None,
               truncate_from: Union[str, int] = 'right',
               ) -> Tuple[List[int], List[int]]:
        """输出文本对应token id和segment id"""
        if isinstance(first_text, str):
            first_tokens = self.tokenize(first_text)
        elif isinstance(first_text, list) and isinstance(first_text[0], str):
            first_tokens = first_text
        else:
            raise ValueError("first_text must be str or list[str], but type {} is given.".format(type(first_text)))

        if second_text is None:
            second_tokens = None
        elif isinstance(second_text, str):
            second_tokens = self.tokenize(second_text)
        elif isinstance(second_text, list) and isinstance(second_text[0], str):
            second_tokens = second_text 
        else:
            raise ValueError("second_text must be None or str or list[str], but type {} is given.".format(type(second_text)))
        
        if max_len is not None:
            if truncate_from == 'right':
                # truncate时每次删除倒数第二个词，因为倒数第一个词是[SEP]
                index = -2
            elif truncate_from == 'left':
                # truncate时每次删除第一个词，因为第零个词是[CLS]
                index = 1
            else:
                # 一般用不到这个，最好别用
                index = truncate_from
            if second_text is not None:
                # len + 1是因为多了一个token [SEP]
                max_len += 1
            truncate_sequences(max_len, index, first_tokens, second_tokens)

        first_token_ids = self.convert_tokens_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_token_ids)

        if second_tokens is not None:
            second_tokens = second_tokens[1:] # 去掉[CLS]
            second_token_ids = self.convert_tokens_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
        
        return first_token_ids, first_segment_ids

    def get_vocab(self):
        return dict(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)
    

class BasicTokenizer(object):
    """基本的分词，包括清理文本，按空格分词，划分开标点符号连接的词，小写化等"""
    def __init__(self, 
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        self.do_lower_case = do_lower_case
        self.never_split = never_split
    
    def tokenize(self, text: str) -> List[str]:
        "将文本切分为token"
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens)) # 这里还要再做一次去除whitespace，有必要吗？
        return output_tokens

    def _run_strip_accents(self, text):
        """去除字符的accents，即去掉字符的变音符号上标，ü --> u"""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn": # 如果该char是变音符号，丢弃
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """
        把标点符号连接的词分开，因为空格没能把它们分开
        如BasicTokenizer._run_split_on_punc("good,but") = ["good", ",", "but"]
        """
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """在汉字两边加空格."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        把文本tokenize为word piece
        使用给定的vocab 进行贪婪最长优先匹配算法(greedy longest-match-first algorithm) 进行 tokenize

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: 已经被`BasicTokenizer`分词后用空格连接的文本

        Returns:
          A list of wordpiece tokens.
        """
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token) # 过长的单词当[UNK]处理
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr  # 非开头的word piece前面需要加##, 以和作为token开头的word piece作区分
                    if substr in self.vocab:  # 当前的substr在vocab中，即它是一个word piece，将其记录保存之后添加到sub_tokens
                        cur_substr = substr
                        break
                    end -= 1  # 没找到，从后缩小substr的搜索范围

                if cur_substr is None:  # 当前token无法被划分成多个word piece
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end  # 从上一找到的substr后面开始继续匹配

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

            return output_tokens


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


if __name__ == "__main__":
    btokenizer = BasicTokenizer(do_lower_case=False)
    tokenizer = Tokenizer(vocab_file="/home/zxy21/codes_and_data/.cache/pretrained_models/bert-base-cased/vocab.txt", do_lower_case=False)
    c_tokenizer = Tokenizer(vocab_file="/home/zxy21/codes_and_data/.cache/pretrained_models/bert-base-multilingual-cased/vocab.txt", do_lower_case=False)

    string = "are you an engineer or data scientist? Do you ship reliable and performant applied machine learning solutions? Check out our Introduction to Keras for engineers.Are you a machine learning researcher? Do you publish at NeurIPS and push the state-of-the-art in CV and NLP? Check out our Introduction to Keras for researchers."
    string2 = "Are you a beginner looking for both an introduction to machine learning and an introduction to Keras and TensorFlow? You're going to need more than a one-pager. And you're in luck: we've got just the book for you."
    chineses_string = "最近，笔者也是花了几个晚上的时间，把656篇长文过了一边，并将其进行了详细的归类划分，主要包括：36篇QA系统（阅读理解、问答、检索）、17篇情感分析（方面级情感分析、篇章集情感分析、情绪分析）、42篇对话系统、45篇信息抽取（关键词抽取、术语抽取、实体抽取、实体分类、关系抽取、事件抽取、观点抽取）、6篇事件检测、68篇预训练语言模型应用（Transformer优化、语言模型下游应用、语言模型探索、分析等）、37篇数据集、任务及评估、45篇机器翻译、37篇多模态、19篇摘要（对话摘要、多文档摘要、代码摘要）、51篇文本生成（段落生成、对话生成、复述、问题生成）、7篇文本风格改写、13篇推理（因果推断、多跳推理、知识推理、常识推理）、21篇模型鲁棒性及对抗、10篇模型压缩（模型优化、剪枝、蒸馏）、19篇小样本（元学习、零样本、低资源）、26篇知识表征、6篇多语言、12篇社会道德伦理偏见、2篇虚假新闻检测、14篇指代、链指、消歧及对齐、3篇ASR、8篇数据增强、2篇纠错、22篇图相关、15篇文本分类、13篇NLP基础（分词、词性、语义理解、句法分析）、60篇其他。"

    basic_token = btokenizer.tokenize(string)
    print("Basic tokens: ", basic_token)
    wordpiece_token = tokenizer(string)
    print("Word piece tokens: ", wordpiece_token)
    print("Chinese Word piece tokens: ", c_tokenizer(chineses_string))
    encoded = tokenizer.encode(first_text=string, second_text=string2, max_len=100)
    print("encoded: ", encoded)
    print(len(encoded[0]))
    print(tokenizer.convert_ids_to_tokens(encoded[0]))

