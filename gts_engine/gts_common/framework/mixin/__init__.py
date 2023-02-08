"""mixin集合.

包含:
    * OptionalLoggerMixin: 支持选择从logger或者print进行信息输出
    * ArgsMixin: 支持与参数集合模块进行交互

另外，Mixin设计模式相关思想，可以参考https://www.zhihu.com/question/20778853
"""
from .args_mixin import ArgsMixin
from .optional_logger_mixin import OptionalLoggerMixin

__all__ = ["OptionalLoggerMixin", "ArgsMixin"]
