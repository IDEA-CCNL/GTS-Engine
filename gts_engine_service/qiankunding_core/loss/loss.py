

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    https://paperswithcode.com/method/focal-loss
    Multi-class Focal loss
    FocalLoss(p_{t}) = -(1 - p_{t})^{\gamma} \log (p_{t})
    Focal loss 主要解决训练时的类别不平衡问题。让模型专注在较难的负例上学习，而给学得已经比较好的例子更小的损失
    它是一种动态缩放的交叉熵损失，缩放因子可以降低已经学得很好的样例对loss的贡献，集中精力在较难的错误样本上学习。
    """
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, weight=self.weight, ignore_index=self.ignore_index)
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Implementation of Label Smoothing Cross Entropy Loss 
    Modified from fastai https://github.com/fastai/fastai2/blob/master/fastai2/layers.py#L338
    """
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        input: [N, C], logit, not probability
        target: [N, ]
        """
        c = output.size(-1)
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        elif self.reduction == 'mean':
            loss = -log_preds.sum(dim=-1).mean()
        else:
            raise ValueError("reduction must be `sum` or `mean` but `{}` is given".format(self.reduction))

        return self.eps * loss / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction, 
                                                                 ignore_index=self.ignore_index)


if __name__ == "__main__":
    a = torch.tensor([[5.12, 4, 3, 2],
                      [1.42, 2, 3, 4]])
    b = torch.tensor([0, 3])
    loss_1 = LabelSmoothingCrossEntropy()
    loss_2 = nn.CrossEntropyLoss()

    loss1 = loss_1(a, b)
    loss2 = loss_2(a, b)
    print(loss1, loss2)
