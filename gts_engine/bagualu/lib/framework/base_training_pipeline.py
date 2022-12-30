from abc import ABCMeta, abstractmethod
from typing import Optional, List

from .mixin import ArgsMixin

class BaseTrainingPipeline(ArgsMixin, metaclass=ABCMeta):
    
    def __init__(self, args_parse_list: Optional[List[str]] = None):
        ArgsMixin.__init__(self, args_parse_list)
    
    @abstractmethod
    def _before_training(self) -> None:
        pass
    
    @abstractmethod
    def _train(self) -> None:
        pass
    
    @abstractmethod
    def _after_training(self) -> None:
        pass
    
    def main(self):
        self._before_training()
        self._train()
        self._after_training()
