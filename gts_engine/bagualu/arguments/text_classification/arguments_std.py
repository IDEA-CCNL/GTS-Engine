from typing import Optional
from pydantic import FilePath
from pathlib import Path

from ...lib.framework.base_arguments import GeneralParser
from ...lib.components.lightning_callbacks.adaptive_val_intervals import (
    ADAPTIVE_VAL_INTERVAL_MODE)
from ...lib.framework.classification_finetune import (
    BaseTrainingArgumentsClf,
    BaseInferenceArgumentsClf)
from ...lib.utils.detect_gpu_memory import detect_gpu_memory, decide_gpu


class TrainingArgumentsClfStd(BaseTrainingArgumentsClf):

    _aug_eda_path: Optional[FilePath]
    precision: int=32
    use_gradient_checkpointing: str="False"

    @property
    def aug_eda_path(self) -> FilePath:
        """eda缓存文件"""
        return self.input_dir / "eda_augment.json" if (
            self._aug_eda_path is None) else self._aug_eda_path

    @property
    def memory_optimization_setting(self) -> None:
        # 基于显卡显存大小，设定显存优化的参数
        gpu_memory, gpu_cur_used_memory = detect_gpu_memory()
        gpu_type = decide_gpu(gpu_memory)
        self.precision = 16 if "low" in gpu_type else 32 
        self.use_gradient_checkpointing = "True" if gpu_type == "lower_gpu" else "False"
        print("基于当前显卡的显存大小 {}M, 设置training precision为{}, 设置use_gradient_checkpointing为{}".format(gpu_memory, self.precision, self.use_gradient_checkpointing))

    def _add_args(self, parser: GeneralParser) -> None:
        super()._add_args(parser)
        parser.add_argument("--aug_eda_path", dest="_aug_eda_path",
                            type=Path, default=None, help="[可选]指定eda文件缓存路径")
        # 不传入时，才为false
        parser.add_argument("--use_knn", dest="use_knn",
                            default=False, action="store_true",
                            help="whether or not to use knn component")
        parser.add_argument("--rdrop_gate", dest="rdrop_gate",
                            type=int, default=50,
                            help="Determine whether to use rdrop based on the \
                            number of categories in the classification task")
        parser.add_argument("--use_rdrop", dest="use_rdrop",
                            default=False, action="store_true",
                            help="whether or not to use rdrop component")
        parser.add_argument("--rdrop_alpha", dest="rdrop_alpha",
                            type=int, default=5)
    aug_eda_gate: bool = True
    dev_resample_thres: int = 1000  # dev数据超过阈值进行重采样
    validation_mode = ADAPTIVE_VAL_INTERVAL_MODE.ADAPTIVE

class InferenceArgumentsClfStd(BaseInferenceArgumentsClf):
    ...
