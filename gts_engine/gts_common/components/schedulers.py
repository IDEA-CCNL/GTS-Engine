"""pytorch schedulers."""
from typing import Callable, List, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import AdamW

Lambda = Union[Callable[[int], float], List[Callable[[int], float]]]


class LambdaLRWithMinLR(LambdaLR):
    """支持设置lr下限的LambdaLR.

    原生LambdaLR不支持设置lr下限，通过继承以支持
    """

    def __init__(self,
                 optimizer: Optimizer,
                 lr_lambda: Lambda,
                 last_epoch: int = -1,
                 min_lr: float = 1e-8,
                 verbose: bool = False):
        """实例化LambdaLRWithMinLR Scheduler.

        Args:
            optimizer (Optimizer): 指定优化器
            lr_lambda (Lambda): 学习率计算函数，为step到学习率的映射
            last_epoch (int, optional): Defaults to -1.
            min_lr (float, optional): 学习率下限。 Defaults to 1e-8.
            verbose (bool, optional):  Defaults to False.
        """
        self._min_lr = min_lr
        super().__init__(optimizer, lr_lambda, last_epoch,
                         verbose)  # (pyi文件和定义冲突)

    def get_lr(self) -> List[float]:
        """获取学习率列表."""
        lr_list: List[float] = super().get_lr()  # (pyi文件和定义冲突)
        return [max(lr, self._min_lr) for lr in lr_list]


def warmup_linear_decay_scheduler_factory(
        optimizer: AdamW,
        warm_up_epoch: int,
        decay_epoch: int,
        epoch: int = 4,
        min_lr: float = 1e-8,
        train_data_length: int = 1000,
        batch_size: int = 8,
        last_step: int = -1) -> LambdaLRWithMinLR:
    """生成warm up scheduler.

    Args:
        optimizer (AdamW): 优化器
        warm_up_epoch (int): warm up epoch数
        decay_epoch (int): decay epoch数
        epoch (int, optional): epoch总数. Defaults to 4.
        min_lr (float, optional): 学习率下限. Defaults to 1e-8.
        train_data_length (int, optional): 训练样本量. Defaults to 1000.
        batch_size (int, optional): 训练batch size. Defaults to 8.
        last_step (int, optional): . Defaults to -1.

    Returns:
        LambdaLRWithMinLR: 生成的warm up scheduler
    """
    epoch_step = train_data_length / batch_size
    warm_up_steps = int(epoch_step * warm_up_epoch)
    decay_steps = int(epoch_step * decay_epoch)
    all_steps = epoch * (train_data_length / batch_size)

    def lr_lambda(current_step: int) -> float:
        if current_step <= warm_up_steps:
            ratio = current_step / warm_up_steps
        elif warm_up_steps < current_step <= decay_steps:
            ratio = 1.0
        else:
            ratio = 1.0 - ((current_step - decay_steps) /
                           (all_steps - decay_steps))
        return ratio

    return LambdaLRWithMinLR(optimizer, lr_lambda, last_step, min_lr)
