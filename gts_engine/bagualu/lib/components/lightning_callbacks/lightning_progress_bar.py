from pytorch_lightning.callbacks.progress.base import ProgressBarBase
from typing import Union
from logging import Logger

class LitProgressBar(ProgressBarBase):
    def __init__(self, logger: Logger):
        super().__init__()  # don't forget this :)
        self.enable = True # type: ignore
        self._logger = logger

    def disable(self):
        self.enable = False # type: ignore

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)
        metrics = self.get_metrics(trainer, pl_module)
        metrics_str_list = [self._get_format(name, val) for name, val in metrics.items()]
        metrics_str_list.append(f"batch {batch_idx}/{self.total_batches_current_epoch}")
        metrics_str = " ".join(metrics_str_list)
        self._logger.info(metrics_str)
        
    
    def _get_format(self, name: str, val: Union[float, int, str]) -> str:
        if name in ("epoch", "step"):
            f = f"{name} {int(val)}"
        elif name in ("lr"):
            f = f"{name} {val:.4e}"
        elif isinstance(val, float):
            f = f"{name} {val:.4f}"
        else:
            f = f"{name} {val}"
        return f