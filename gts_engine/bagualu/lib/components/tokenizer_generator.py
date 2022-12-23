import os
from typing import Dict, Union
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from types import MethodType
from pydantic import DirectoryPath

from ..framework.consts import RUN_MODE
from ..utils.json_processor import load_json


class TokenizerGenerator:
    """Tokenizer生成器
        
        通过generate_tokenizer()方法，根据预训练模型路径生成相应的tokenizer
    """
    @classmethod
    def generate_tokenizer(cls, pretrained_model_dir: Union[str, DirectoryPath]) -> PreTrainedTokenizer:
        if not os.path.exists(pretrained_model_dir):
            raise Exception("pretrained model directory dose not exists")
        pretrained_model_config_path = os.path.join(pretrained_model_dir, 'config.json')
        if not os.path.exists(pretrained_model_config_path):
            raise Exception(f"model config file {pretrained_model_config_path} does not exist")
        model_config: Dict = load_json(pretrained_model_config_path)
        if "vocab_size" not in model_config:
            raise Exception(f"model config does not have field: 'vocab_size'")
        return cls.__get_ernie_tokenizer(pretrained_model_dir) if model_config['vocab_size'] == 18000 \
            else cls.__get_bert_tokenizer(pretrained_model_dir)
            
    @classmethod
    def __get_ernie_tokenizer(cls, pretrained_model_dir: Union[str, DirectoryPath]) -> PreTrainedTokenizer:
        """
        load ernie tokenizer
        """
        added_token = ['[unused'+str(i+1)+']' for i in range(36)]
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,
                            additional_special_tokens=added_token)
        return cls.__tokenizer_wraper(tokenizer)
    
    @classmethod
    def __get_bert_tokenizer(cls, pretrained_model_dir: Union[str, DirectoryPath]) -> PreTrainedTokenizer:
        """
        load normal tokenizer
        """
        added_token = ['[unused'+str(i+1)+']' for i in range(99)]
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,
                                        additional_special_tokens=added_token)
        return cls.__tokenizer_wraper(tokenizer)

    @classmethod
    def __tokenizer_wraper(cls, tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
        """
            原tokenizer的get_vocab方法运算较慢，覆盖tokenizer实例的get_vocab方法，使之只需访问计算好的缓存
        """
        vocab = tokenizer.get_vocab()
        setattr(tokenizer, "vocab_cache", vocab)
        def get_vocab(self: PreTrainedTokenizer):
            return getattr(self, "vocab_cache")
        tokenizer.get_vocab = MethodType(get_vocab, tokenizer)
        return tokenizer


