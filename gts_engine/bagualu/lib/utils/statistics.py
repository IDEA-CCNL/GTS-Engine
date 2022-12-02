from typing import Hashable, List, TypeVar, Sequence, Literal, Union, Iterable, Callable, Generic, Optional
import numpy as np
from functools import cmp_to_key


def interval_mean(list_: List, low_quantile=0.25, high_quantile=0.75) -> float:
    """计算上下分位数之间元素的均值"""
    length = len(list_)
    low_idx = int(length * low_quantile)
    high_idx = int(length * high_quantile)
    res = np.mean(np.sort(list_)[low_idx:high_idx])
    return res # type: ignore

T = TypeVar(name="T", bound=Hashable)
def acc(predict_list: List[T], true_list: List[T]) -> float:
    assert len(predict_list) == len(true_list)
    return np.mean(np.array(predict_list) == np.array(true_list)) # type: ignore



class ExponentialSmoothingList:
    
    number = Union[float, int]
    
    def __init__(self, level: Literal[1, 2, 3], alpha: Union[float, List[float]] ,warmup_steps: int = 3) -> None:
        self.__check_condition(level, alpha)
        if isinstance(alpha, list):
            self.__alpha_list = alpha
        else:
            self.__alpha_list = [alpha] * level
        self.__values = [[] for _ in range(level + 1)]
        self.__level = level
        self.__warmup_steps = warmup_steps
        
    def append(self, value: Union[number, Sequence[number]]):
        if isinstance(value, Iterable):
            for v in value:
                self.__step_value(v)
        else:
            self.__step_value(value)
            
    @property
    def values(self):
        return self.__values
    
    def __repr__(self) -> str:
        return str(self.__values)
    
    def __getitem__(self, level: int):
        return self.__values[level]
    
    def __len__(self):
        return len(self.__values[0])
    
    #############################################################################################
    ######################################## private ##########################################
        
    def __step_value(self, value: number) -> None:
        self.__values[0].append(value)
        for level in range(1, self.__level + 1):
            self.__append_level_value_list(level)
    
    def __append_level_value_list(self, level: int):
        if len(self.__values[level - 1]) <= self.__warmup_steps:
            self.__values[level].append(self.__values[level - 1][-1])
        else:
            alpha = self.__alpha_list[level - 1]
            if len(self.__values[level - 1]) == self.__warmup_steps + 1:
                self.__re_calculate_warmup_steps(level, alpha)
            smoothed_value = alpha * self.__values[level][-1] + (1 - alpha) * self.__values[level - 1][-1]
            self.__values[level].append(smoothed_value)
    
    def __re_calculate_warmup_steps(self, level: int, alpha: float):
        self.__values[level][0] = sum(self.__values[level - 1][:self.__warmup_steps]) / self.__warmup_steps
        for step in range(1, self.__warmup_steps):
            smoothed_value = alpha * self.__values[level][step - 1] + (1 - alpha) * self.__values[level - 1][step]
            self.__values[level][step] = smoothed_value
    
    def __check_condition(self, level, alpha) -> None:
        if isinstance(alpha, list):
            try:
                assert len(alpha) == level
            except:
                raise Exception("the alpha list should be as long as level")
            for a in alpha:
                try:
                    assert a >= 0 and a <= 1
                except:
                    raise Exception("alpha should be in [0, 1]")
        else:
            try:
                assert alpha >= 0 and alpha <= 1
            except:
                raise Exception("alpha should be in [0, 1]")


_MaxVar = TypeVar("_MaxVar")         
class DynamicMax(Generic[_MaxVar]):
    """带缓存、可自定义比较符的、可指定top n的动态max类"""
    
    def __init__(
        self, 
        top_n: int = 1, 
        allow_repeat: bool = True,
        cmp: Callable[[_MaxVar, _MaxVar], Union[int, float]] = lambda x1, x2: x1 - x2, # type: ignore
    ):
        self.__cmp = cmp
        self.__top_n = top_n
        self.__allow_repeat = allow_repeat
        self.__cache: List[_MaxVar] = []
    
    def step(self, value: Union[_MaxVar, Sequence[_MaxVar]]) -> None:
        if isinstance(value, Iterable):
            for v in value:
                self.__step_value(v)
        else:
            self.__step_value(value)
        
    def __step_value(self, value: _MaxVar) -> None:
        self.__cache.append(value)
        if not self.__allow_repeat:
            self.__cache = list(set(self.__cache))
        self.__cache.sort(key=cmp_to_key(self.__cmp), reverse=True) # type: ignore
        if len(self.__cache) > self.__top_n:
            self.__cache.pop()
            
    @property
    def max_list(self) -> List[_MaxVar]:
        return self.__cache
        
    @property
    def max(self) -> _MaxVar:
        return self.__cache[-1]
    

    
    