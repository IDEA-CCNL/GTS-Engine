# coding=utf-8
# Copyright 2021 The IDEA Authors. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=no-member

from typing import Optional
import torch
from torch import Tensor


class NegativeSampleLoss(torch.nn.Module):
    """ 负例采样损失 """
    def __init__(self, p=0.4):
        super().__init__()
        self.bce_loss = torch.nn.BCELoss(reduction='none')
        self.dropout = torch.nn.Dropout(p=p)

    def forward(self,
                logits: Tensor,
                labels: Tensor,
                mask: Tensor = None) -> Tensor:
        """ forward """
        loss = self.bce_loss(logits, labels)
        loss_pos = loss * labels.float()
        loss_neg = loss * (1.0 - labels.float())
        loss_neg = self.dropout(loss_neg)
        loss = loss_pos + loss_neg
        if mask is not None:
            assert mask.shape == loss.shape
            loss = loss * mask
            return torch.sum(loss) / torch.sum(mask)
        mean_loss = torch.mean(loss)
        return mean_loss


class DiceLoss(torch.nn.Module):
    """
    REF https://github.com/ShannonAI/dice_loss_for_NLP/blob/master/loss/dice_loss.py
    Dice coefficient for short, is an F1-oriented statistic used
    to gauge the similarity of two sets.
    Given two sets A and B, the vanilla dice coefficient between them is given as follows:
        Dice(A, B)  = 2 * True_Positive / (2 * True_Positive + False_Positive + False_Negative)
                    = 2 * |A and B| / (|A| + |B|)
    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional):
            [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional):
            [True, False], specifies whether the input tensor is normalized
            by Sigmoid/Softmax funcs.
        ohem_ratio: max ratio of positive/negative, defautls to 0.0, which means no ohem.
        alpha: dsc alpha
    Shape:
        - input: (*)
        - target: (*)
        - mask: (*) 0,1 mask for the input sequence.
        - Output: Scalar loss
    """
    def __init__(self,
                 smooth: Optional[float] = 1,
                 square_denominator: Optional[bool] = True,
                 with_logits: Optional[bool] = False,
                 ohem_ratio: float = 1.,
                 alpha: float = 0.01,
                 reduction: Optional[str] = "mean") -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratio = ohem_ratio
        self.alpha = alpha

    def forward(self, inputs: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """ forward """
        loss = self._binary_class(inputs, target, mask=mask)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _compute_dice_loss(self, flat_input, flat_target):
        # flat_input = ((1 - flat_input) ** self.alpha) * flat_input
        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            denominator = flat_input.sum() + flat_target.sum() + self.smooth
        else:
            flat_input = torch.square(flat_input)
            flat_target = torch.square(flat_target)
            denominator = torch.sum(flat_input, -1) + torch.sum(flat_target, -1) + self.smooth

        loss = 1 - ((2 * interection + self.smooth) / denominator)

        return loss

    def _binary_class(self, inputs, target, mask=None):
        # flat_input = inputs.view(-1)
        # flat_target = target.view(-1).float()
        # mask = mask.view(-1).float()
        flat_input = inputs.reshape(-1)
        flat_target = target.reshape(-1).float()
        mask = mask.reshape(-1).float()
        flat_input = torch.sigmoid(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        if self.ohem_ratio > 0:
            pos_example = target > 0.5
            neg_example = target <= 0.5
            mask_neg_num = mask <= 0.5

            pos_num = pos_example.sum() - (pos_example & mask_neg_num).sum()
            neg_num = neg_example.sum()
            keep_num = min(int(pos_num * self.ohem_ratio), neg_num) + 1

            neg_scores = torch.masked_select(flat_input, neg_example.bool())
            neg_scores_sort, _ = torch.sort(neg_scores, )
            threshold = neg_scores_sort[-keep_num+1]
            cond = (flat_input >= threshold) | pos_example.view(-1)
            ohem_mask = torch.where(cond, 1, 0)

            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)


class DecoupledBCEloss(torch.nn.Module):
    """ Decoupled BCE Loss

    Args:
        threshold (float, optional): threshold. Defaults to 0.1
        alpha (float, optional): alpha for positive loss. Defaults to 1.
        beta (float, optional): beta for negative loss. Defaults to 100.
    """
    def __init__(self,
                 threshold: float = 0.1,
                 alpha: float = 1.,
                 beta: float = 100.) -> None:
        super().__init__()

        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta

        self._bce_sum = torch.nn.BCELoss(reduction="sum")

    def forward(self,
                source: Tensor,
                target: Tensor,
                mask: Tensor = None) -> Tensor:
        """ forward

        Args:
            source (Tensor): source
            target (Tensor): target
            mask (Tensor): mask. Defaults to None.

        Returns:
            Tensor: loss
        """
        # mask
        if mask is None:
            mask = torch.ones_like(target)
        else:
            source = source * mask
            target = target * mask

        ones = torch.ones_like(source)
        zeros = torch.zeros_like(source)

        # source和target中超过阈值的位置
        source_pos = torch.where(source < self.threshold, zeros, ones) * mask
        target_pos = torch.where(target < self.threshold, zeros, ones) * mask

        # source和target中任一超过阈值(pos)和全部低于阈值(neg)的位置
        mask_pos = torch.where(source_pos + target_pos > 0, ones, zeros) * mask
        mask_neg = torch.where(source_pos + target_pos <= 0, ones, zeros) * mask

        # pos和neg的数量
        num_pos = mask_pos.sum()
        num_neg = mask_neg.sum()

        # pos部分的loss
        if num_pos <= 0: # 没有pos
            loss_pos = 0.
        else:
            loss_pos = self._bce_sum(source * mask_pos, target * mask_pos) / num_pos

        # neg部分的loss
        if num_neg <= 0: # 没有neg
            loss_neg = 0.
        else:
            loss_neg = self._bce_sum(source * mask_neg, target * mask_neg) / num_neg

        loss = self.alpha * loss_pos + self.beta * loss_neg

        return loss


class DecoupledMSEloss(torch.nn.Module):
    """ Decoupled MSE Loss

    Args:
        threshold (float, optional): threshold. Defaults to 0.1
        alpha (float, optional): alpha for positive loss. Defaults to 1.
        beta (float, optional): beta for negative loss. Defaults to 100.
    """
    def __init__(self,
                 threshold: float = 0.1,
                 alpha: float = 1.,
                 beta: float = 100.) -> None:
        super().__init__()

        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta

        self._mse_sum = torch.nn.MSELoss(reduction="sum")

    def forward(self,
                source: Tensor,
                target: Tensor,
                mask: Tensor = None) -> Tensor:
        """ forward

        Args:
            source (Tensor): source
            target (Tensor): target
            mask (Tensor): mask. Defaults to None.

        Returns:
            Tensor: loss
        """
        # mask
        if mask is not None:
            source = source * mask
            target = target * mask

        ones = torch.ones_like(source)
        zeros = torch.zeros_like(source)

        # source和target中超过阈值的位置
        source_pos = torch.where(source < self.threshold, zeros, ones) * mask
        target_pos = torch.where(target < self.threshold, zeros, ones) * mask

        # source和target中任一超过阈值(pos)和全部低于阈值(neg)的位置
        mask_pos = torch.where(source_pos + target_pos > 0, ones, zeros) * mask
        mask_neg = torch.where(source_pos + target_pos <= 0, ones, zeros) * mask

        # pos和neg的数量
        num_pos = mask_pos.sum()
        num_neg = mask_neg.sum()

        # pos部分的loss
        if num_pos <= 0: # 没有pos
            loss_pos = 0.
        else:
            loss_pos = self._mse_sum(source * mask_pos, target * mask_pos) / num_pos

        # neg部分的loss
        if num_neg <= 0: # 没有neg
            loss_neg = 0.
        else:
            loss_neg = self._mse_sum(source * mask_neg, target * mask_neg) / num_neg

        loss = self.alpha * loss_pos + self.beta * loss_neg

        return loss


class DistillSelfLoss(torch.nn.Module):
    """ Distill Self Loss """
    def __init__(self):
        super().__init__()
        self.d_mse = DecoupledMSEloss(threshold=0.1,
                                    alpha=1.,
                                    beta=100.)

    def forward(self,
                s_logits: Tensor,
                t_logits: Tensor,
                mask: Tensor) -> Tensor:
        """ forward

        Args:
            s_logits (Tensor): source logits, (bsz, seq, seq, label)
            t_logits (Tensor): target logits, (bsz, seq, seq, label)
            mask (Tensor): mask, (bsz, seq, seq)

        Returns:
            Tensor: loss
        """
        loss = self.d_mse(s_logits, t_logits, mask)
        return loss
