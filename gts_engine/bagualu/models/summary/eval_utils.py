from typing import Any, Dict, List

import torch
from torch import nn
from torchmetrics.text.rouge import ROUGEScore

from .tokenizers_pegasus import PegasusTokenizer


class SummarizationEvaluation(nn.Module):

    def __init__(self, pretrained_model_dir):
        super().__init__()

        self._tokenizer = PegasusTokenizer.from_pretrained(
            pretrained_model_dir)

    def forward(self, generated_ids, target_ids):

        preds = self._tokenizer.batch_decode(generated_ids,
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
        labels = torch.where(target_ids != -100, target_ids,
                             self._tokenizer.pad_token_id)
        labels = self._tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)

        tokenize_preds = [chinese_char_tokenize(p) for p in preds]
        tokenize_labels = [chinese_char_tokenize(label) for label in labels]

        return tokenize_preds, tokenize_labels


def chinese_char_tokenize(line):
    line = line.strip()
    line_in_chars = []

    for char in line:
        if _is_chinese_char(ord(char)):
            line_in_chars.append(" ")
            line_in_chars.append(char)
            line_in_chars.append(" ")
        else:
            line_in_chars.append(char)

    return "".join(line_in_chars)


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)):
        return True

    return False


def get_summarization_report(y_true: List[str], y_pred: List[str],
                             rouge_keys: str) -> Dict[str, Any]:

    rouge_keys_ = tuple(rouge_keys.split(','))

    rouge_metric = ROUGEScore(rouge_keys=rouge_keys_, normalizer=lambda x: x)

    rouge_dict = rouge_metric(y_pred, y_true)

    for k, v in rouge_dict.items():
        rouge_dict[k] = v.item()

    return rouge_dict
