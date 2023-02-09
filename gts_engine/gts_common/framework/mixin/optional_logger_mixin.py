"""提供可选使用logger或print输出功能的Mixin模块."""
import logging
from typing import Optional


class OptionalLoggerMixin:
    """提供可选使用logger或print输出功能的Mixin.

    类实例工作时需要输出信息，为了同时兼容使用logger和print进行输出，此Mixin提供了info()
    方法，当类实例化时传入logger_name时，通过该logger_name对应的logger进行输出，否则通过
    print进行输出。

    Example:
        >>> class HasOptionalLogger(..., OptionalLoggerMixin):
        ...
        ...     def __init__(self, ..., logger_name: Optional[str] = None):
        ...         OptionalLoggerMixin.__init__(self, logger_name)  # 实例化OptionalLoggerMixin
        ...         ...
        ...
        >>> use_logger = HasOptionalLogger(logger_name="name")
        >>> use_logger.info("output via logger 'name'")
        >>> user_print = HasOptionalLogger(logger_name=None)
        >>> user_print.info("output via print")
    """

    def __init__(self, logger_name: Optional[str] = None):
        """实例化OptionalLoggerMixin.

        Args:
            logger_name (Optional[str], optional):
                输出的logger全局名称，为None时则使用print输出。Defaults to None.
        """
        self._logger = logging.getLogger(logger_name) if logger_name else None

    def info(self, info: str):
        """使用logger.info()或print()打印信息."""
        if hasattr(self, "_logger") and self._logger:
            self._logger.info(info)
        else:
            print(info)
