from typing import Optional
import logging

class OptionalLoggerMixin:
    """支持选择性输出到logger或print的mixin"""
    def __init__(self, logger_name: Optional[str] = None):
        self._logger = logging.getLogger(logger_name) if logger_name else None
    
    def info(self, info: str):
        if hasattr(self, "_logger") and self._logger:
            self._logger.info(info)
        else:
            print(info)