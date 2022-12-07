from abc import ABCMeta, abstractmethod
from typing import Optional, List, Sequence, Dict
from pydantic import FilePath, BaseModel

from .mixin import ArgsMixin

InfSample = BaseModel
InfOutput = Dict

class BaseInferenceManager(ArgsMixin, metaclass=ABCMeta):
    
    def __init__(self, args_parse_list: Optional[List[str]] = None):
        ArgsMixin.__init__(self, args_parse_list)
    
    @abstractmethod
    def prepare_inference(self) -> None: ...
    
    @abstractmethod
    def inference(self, sample: Sequence[InfSample]) -> InfOutput: ...