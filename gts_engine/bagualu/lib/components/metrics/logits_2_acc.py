import torch
from torch import nn, Tensor


class Logits2Acc(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.__softmax = torch.nn.Softmax(dim=-1)

    def forward(self, logits: Tensor, labels: Tensor):
        prob = self.__softmax(logits)
        y_pred = torch.argmax(prob, dim=-1)

        y_pred = y_pred.view(size=(-1, ))
        y_true = labels.view(size=(-1, ))

        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())

        return acc, y_pred