"""loss计算工具集

Todo:
    - [] (Jiang Yuzhen) 待修改和补全(详见components.__init__.py)
"""
from .max_multi_logits import MaxMultiLogits
from .label_smoothing import LabelSmoothing
from .rdrop_kl_loss import compute_kl_loss
from .ie_losses import (
    NegativeSampleLoss,
    DiceLoss,
    DecoupledBCEloss,
    DecoupledMSEloss,
    DistillSelfLoss
)

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
