from transformers.optimization import AdamW
from torch.optim import Optimizer
from abc import ABCMeta, abstractmethod
from torch.optim.lr_scheduler import LambdaLR
from typing import Callable, Union, List

Lambda = Union[Callable[[int], float], List[Callable[[int], float]]]
class LambdaLRWithMinLR(LambdaLR):
    """重写LambdaLR，支持设置lr下限"""
    def __init__(self, optimizer: Optimizer, lr_lambda: Lambda, last_epoch: int=-1, min_lr: float=1e-8, verbose: bool=False):
        self._min_lr = min_lr
        super().__init__(optimizer, lr_lambda, last_epoch, verbose) # type: ignore # pyi文件和定义冲突
    
    def get_lr(self):
        lr_list: List[float] = super().get_lr() # type: ignore # pyi文件和定义冲突
        return [max(lr, self._min_lr) for lr in lr_list]
        
def warmup_linear_decay_scheduler_factory(
    optimizer: AdamW, 
    warm_up_epoch: int, 
    decay_epoch: int, 
    epoch: int=4, 
    min_lr: float=1e-8,
    train_data_length: int=1000,
    batch_size: int=8,
    last_step: int=-1
) -> LambdaLRWithMinLR:
    
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
            ratio = 1.0 - ((current_step - decay_steps) / (all_steps - decay_steps))
        return ratio   
     
    return LambdaLRWithMinLR(optimizer, lr_lambda, last_step, min_lr)


