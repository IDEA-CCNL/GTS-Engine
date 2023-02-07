"""loss计算工具集.

Todo:
    - [] (Jiang Yuzhen) 待修改和补全(详见components.__init__.py)
"""
from .ie_losses import (DecoupledBCEloss, DecoupledMSEloss, DiceLoss,
                        DistillSelfLoss, NegativeSampleLoss)
from .label_smoothing import LabelSmoothing
from .max_multi_logits import MaxMultiLogits
from .rdrop_kl_loss import compute_kl_loss

__all__ = [
    "MaxMultiLogits",
    "LabelSmoothing",
    "compute_kl_loss",
    "NegativeSampleLoss",
    "DiceLoss",
    "DecoupledBCEloss",
    "DecoupledMSEloss",
    "DistillSelfLoss",
]
