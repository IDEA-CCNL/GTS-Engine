"""Tokenizer生成器模块."""
import os
from types import MethodType
from typing import Dict, Union

from gts_common.utils.json_utils import load_json
from pydantic import DirectoryPath
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


class TokenizerGenerator:
    """Tokenizer生成器.

    通过generate_tokenizer()方法，根据预训练模型路径生成相应的tokenizer
    """

    @classmethod
    def generate_tokenizer(
        cls,
        pretrained_model_dir: Union[str,
                                    DirectoryPath]) -> PreTrainedTokenizer:
        """根据预训练模型路径生成tokenizer.

        Args:
            pretrained_model_dir (Union[str, DirectoryPath]): 预训练模型路径

        Raises:
            Exception: 预训练模型不存在
            Exception: 预训练模型路径不存在config.json文件
            Exception: config.json不存在"vocab_size"字段

        Returns:
            PreTrainedTokenizer: 预训练模型对应的tokenizer
        """
        if not os.path.exists(pretrained_model_dir):
            raise Exception("pretrained model directory dose not exists")
        pretrained_model_config_path = os.path.join(pretrained_model_dir,
                                                    'config.json')
        if not os.path.exists(pretrained_model_config_path):
            raise Exception(
                f"model config file {pretrained_model_config_path} "
                "does not exist")
        model_config: Dict = load_json(pretrained_model_config_path)
        if "vocab_size" not in model_config:
            raise Exception("model config does not have field: 'vocab_size'")
        return (cls.__get_ernie_tokenizer(pretrained_model_dir)
                if model_config['vocab_size'] == 18000 else
                cls.__get_bert_tokenizer(pretrained_model_dir))

    @classmethod
    def __get_ernie_tokenizer(
        cls,
        pretrained_model_dir: Union[str,
                                    DirectoryPath]) -> PreTrainedTokenizer:
        """load ernie tokenizer."""
        added_token = ['[unused' + str(i + 1) + ']' for i in range(36)]
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_dir, dditional_special_tokens=added_token)
        return cls.__tokenizer_wrapper(tokenizer)

    @classmethod
    def __get_bert_tokenizer(
        cls,
        pretrained_model_dir: Union[str,
                                    DirectoryPath]) -> PreTrainedTokenizer:
        """load normal tokenizer."""
        added_token = ['[unused' + str(i + 1) + ']' for i in range(99)]
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_dir, additional_special_tokens=added_token)
        return cls.__tokenizer_wrapper(tokenizer)

    @classmethod
    def __tokenizer_wrapper(
            cls, tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
        """原tokenizer的get_vocab方法运算较慢，覆盖tokenizer实例的get_vocab方法，
        使之只需访问计算好的缓存."""
        # 计算vocab并写入缓存
        vocab = tokenizer.get_vocab()
        setattr(tokenizer, "vocab_cache", vocab)

        # 新的get_vocab函数，使之访问缓存而不是动态计算
        def get_vocab(self: PreTrainedTokenizer):
            return getattr(self, "vocab_cache")

        # 将定义的方法注入tokenizer对象
        tokenizer.get_vocab = MethodType(get_vocab, tokenizer)
        return tokenizer
