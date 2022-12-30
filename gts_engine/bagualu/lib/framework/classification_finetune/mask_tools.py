import numpy as np
from typing import List, NamedTuple
from transformers.tokenization_utils import PreTrainedTokenizer
import re

from .prompt import PromptBase

class MaskedLmInstance(NamedTuple):
    index: int
    label: str

class CreateMaskedLmPredictionsOutput(NamedTuple):
    output_tokens: List[str]
    masked_lm_positions: List[int]
    masked_lm_labels: List[str]
    

def create_masked_lm_predictions(tokens: List[str],
                                 masked_lm_prob: float,
                                 max_predictions_per_seq: int,
                                 vocab_words: List[str],
                                 index: int,
                                 masked_label: str='[MASK]',
                                 do_wwm: bool=True):
    """Creates the predictions for the masked LM objective."""

    rng = np.random.RandomState(seed=((1234 + index) % 2 ** 32))

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]" or token == '[MASK]':
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_wwm and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    # if non chinese is False, that means it is chinese, then try to remove "##" which is added previously
    output_tokens = [t[2:] if len(re.findall(
        '##[\u4E00-\u9FA5]', t)) > 0 else t for t in tokens]  # 去掉"##"
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms: List[MaskedLmInstance] = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = masked_label
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index][2:] if len(re.findall('##[\u4E00-\u9FA5]', tokens[index])) > 0 \
                        else tokens[index]  # 去掉"##"
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(
                        0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(
                index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions: List[int] = []
    masked_lm_labels: List[str] = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return CreateMaskedLmPredictionsOutput(output_tokens, masked_lm_positions, masked_lm_labels)

class WWMMaskingOutput(NamedTuple):
    input_ids: List[int]
    mlm_labels: List[int]
    input_mask: List[int]
    masked_lm_positions: List[int]

def wwm_masking(new_tokens: List[str], index: int, tokenizer: PreTrainedTokenizer, prompt_mode: PromptBase, max_length: int, mask_rate: float, offset: int=0):
    output_tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(new_tokens[offset:],
                                                                                        masked_lm_prob=mask_rate,
                                                                                        max_predictions_per_seq=30,
                                                                                        masked_label=prompt_mode.mask_token,
                                                                                        vocab_words=list(tokenizer.get_vocab().keys()),
                                                                                        index=index)

    masked_lm_positions = [ele + offset for ele in masked_lm_positions]
    input_ids = tokenizer.convert_tokens_to_ids(
        new_tokens[:offset] + output_tokens)
    assert isinstance(input_ids, list) and len(input_ids) <= max_length
    input_mask = [1] * len(input_ids)
    assert len(input_mask) <= max_length
    while len(input_ids) < max_length:
        input_ids.append(0)
        input_mask.append(0)
    assert len(input_ids) == max_length
    assert len(input_mask) == max_length

    masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
    mlm_labels = [-100] * max_length
    assert isinstance(masked_lm_ids, list)
    for pos, masked_id in zip(masked_lm_positions, masked_lm_ids):
        mlm_labels[pos] = masked_id

    return WWMMaskingOutput(input_ids, mlm_labels, input_mask, masked_lm_positions)


# TODO: 存在问题，之后再来改
# def random_masking(token_ids, max_length, mask_rate, prompt_mode, tokenizer=None):
#     """
#     对输入进行随机mask
#     """
#     rands = np.random.random(len(token_ids))
#     source, target, mask_pos = [], [], []
#     #mask_token = [prompt_mode.mask_token]
#     #mask_id = tokenizer.convert_tokens_to_ids(mask_token)[0]

#     # 删除-CLS SEP id
#     mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
#     cls_id = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
#     sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

#     for i, (r, t) in enumerate(zip(rands, token_ids)):
#         if t == cls_id or t == sep_id:
#             source.append(t)
#             target.append(-100)
#             continue

#         if r < mask_rate * 0.8:
#             source.append(mask_id)
#             target.append(t)
#             mask_pos.append(i)
#         elif r < mask_rate * 0.9:
#             source.append(t)
#             target.append(t)
#             mask_pos.append(i)
#         elif r < mask_rate:
#             source.append(np.random.choice(len(tokenizer.get_vocab().keys()) - 2) + 1)
#             target.append(t)
#             mask_pos.append(i)
#         else:
#             source.append(t)
#             target.append(-100)
    
#     while len(source) < max_length:
#         source.append(0)
#         target.append(-100)
#     return source[:max_length], target[:max_length], mask_pos