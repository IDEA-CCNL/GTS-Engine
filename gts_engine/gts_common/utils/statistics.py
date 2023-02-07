"""统计相关工具集.

Todo:
    - [ ] (Jiang Yuzhen) 将模块名改为statistics_utils或类似名称，和json_utils等工具集保持一致
"""
from functools import cmp_to_key
from typing import (Callable, Generic, Hashable, Iterable, List, Literal,
                    Sequence, TypeVar, Union)

import numpy as np


def interval_mean(list_: List[float],
                  low_quantile: float = 0.25,
                  high_quantile: float = 0.75) -> float:
    """计算上下分位数之间元素的均值."""
    length = len(list_)
    low_idx = int(length * low_quantile)
    high_idx = int(length * high_quantile)
    res: float = np.mean(np.sort(list_)[low_idx:high_idx])  # type: ignore
    return res


_ComparableType = TypeVar(name="_ComparableType", bound=Hashable)


def acc(predict_list: List[_ComparableType],
        true_list: List[_ComparableType]) -> float:
    """计算两个等长序列对应位置元素相等的比例（accuracy），支持所有可比较的元素类型."""
    assert len(predict_list) == len(true_list)
    return np.mean(
        np.array(predict_list) == np.array(true_list))  # type: ignore


class ExponentialSmoothingList:
    """动态计算的支持多阶的指数平滑列表.

    通过`append()`方法更新列表的元素，列表会自动根据历史数据计算每个元素的各阶指数平滑，
    支持使用索引`exp_smooth_list[i]`访问第i阶平滑列表，`exp_smooth_list[i][j]`则为
    第i阶平滑的第j项元素（第0阶代表无指数平滑的原列表）。

    Example:
        >>> smooth_list = ExponentialSmoothingList(3, 0.3, 5)  # 定义3阶平滑、平滑指数为0.3、warmup为5步的指数平滑列表
        >>> smooth_list.append([4, 1, 9, 10, 7, 8, 10, 15])  # 更新列表
        >>> smooth_list
        [[4, 1, 9, 10, 7, 8, 10, 15], [6.2, 2.5599999999999996, 7.068, 9.1204, 7.63612, 7.890836, 9.3672508, 13.31017524], [6.516903999999999, 3.7470711999999993, 6.07172136, 8.205796408, 7.8070229224, 7.865692076719999, 8.916783183016001, 11.9921576229048], [6.4697031780800005, 4.563860793423999, 5.619363190027199, 7.429866442608159, 7.693875978462447, 7.814147247242733, 8.58599240228402, 10.970308056718565]]
        >>> smooth_list[3][-1]  # 获取三阶平滑的最后一项
        10.970308056718565
    """

    def __init__(self,
                 level: Literal[1, 2, 3],
                 alpha: Union[float, List[float]],
                 warmup_steps: int = 3) -> None:
        """实例化指数平滑列表.

        Args:
            level (Literal[1, 2, 3]):
                指数平滑阶数，支持0到3阶
            alpha (Union[float, List[float]]):
                平滑指数。如果为一个List[float]，则为每一阶分别指定对应的平滑指数；
                如果为一个float，则所有阶使用统一的平滑指数。
            warmup_steps (int, optional):
                warmup步数，使平滑列表的初始值为前warmup步数值的均值。
        """
        # 检查alpha是否合规
        self.__check_conditions(level, alpha)
        if isinstance(alpha, list):
            self.__alpha_list = alpha
        else:
            self.__alpha_list = [alpha] * level
        self.__values = [[] for _ in range(level + 1)]
        self.__level = level
        self.__warmup_steps = warmup_steps

    def append(self, value: Union[float, Sequence[float]]):
        """更新指数平滑列表.

        Args:
            value (Union[float, Sequence[float]]): 更新的值或多个值的列表
        """
        if isinstance(value, Iterable):
            for v in value:
                self.__step_value(v)
        else:
            self.__step_value(value)

    @property
    def values(self):
        """访问所有阶的平滑列表."""
        return self.__values

    def __repr__(self) -> str:
        """支持打印."""
        return str(self.__values)

    def __getitem__(self, level: int):
        """支持索引访问第i阶平滑列表."""
        return self.__values[level]

    def __len__(self):
        """支持len()方法获取列表长度."""
        return len(self.__values[0])

    # ========================== privates ===============================

    def __step_value(self, value: float) -> None:
        self.__values[0].append(value)
        for level in range(1, self.__level + 1):
            self.__append_level_value_list(level)

    def __append_level_value_list(self, level: int):
        if len(self.__values[level - 1]) <= self.__warmup_steps:
            # 长度未达到warmup步数时，不做平滑
            self.__values[level].append(self.__values[level - 1][-1])
        else:
            # 长度达到warmup步数，重新计算初始值
            alpha = self.__alpha_list[level - 1]
            if len(self.__values[level - 1]) == self.__warmup_steps + 1:
                self.__re_calculate_warmup_steps(level, alpha)
            smoothed_value = (alpha * self.__values[level][-1] +
                              (1 - alpha) * self.__values[level - 1][-1])
            self.__values[level].append(smoothed_value)

    def __re_calculate_warmup_steps(self, level: int, alpha: float):
        self.__values[level][0] = (
            sum(self.__values[level - 1][:self.__warmup_steps]) /
            self.__warmup_steps)
        for step in range(1, self.__warmup_steps):
            smoothed_value = (alpha * self.__values[level][step - 1] +
                              (1 - alpha) * self.__values[level - 1][step])
            self.__values[level][step] = smoothed_value

    def __check_conditions(self, level, alpha) -> None:
        if isinstance(alpha, list):
            try:
                assert len(alpha) == level
            except Exception:
                raise Exception("the alpha list should be as long as level")
            for a in alpha:
                try:
                    assert a >= 0 and a <= 1
                except Exception:
                    raise Exception("alpha should be in [0, 1]")
        else:
            try:
                assert alpha >= 0 and alpha <= 1
            except Exception:
                raise Exception("alpha should be in [0, 1]")


_T = TypeVar("_T")


class DynamicMax(Generic[_T]):
    """带缓存、可自定义比较符的、记录第top n的动态max类.

    Example:
        >>> # 记录第三长的字符串，重复值只记录一次
        >>> str_len_dyn_max = DynamicMax[str](top_n=3, allow_repeat=False, cmp=lambda s1, s2: len(s1) - len(s2))
        >>> str_len_dyn_max.step(["123456789", "12345", "12", "1234567", "1234", "123456789"])
        >>> str_len_dyn_max.max  # 第三长的字符串
        '12345'
        >>> str_len_dyn_max.max_list
        ['123456789', '1234567', '12345']
        >>> str_len_dyn_max.step("1234567890123")
        >>> str_len_dyn_max.max
        '1234567'
        >>> str_len_dyn_max.max_list
        ['1234567890123', '123456789', '1234567']
    """

    def __init__(
        self,
        top_n: int = 1,
        allow_repeat: bool = True,
        cmp: Callable[[_T, _T],
                      Union[int,
                            float]] = lambda x1, x2: x1 - x2  # type: ignore
    ):
        """实例化.

        Args:
            top_n (int, optional):
                记录最大的n个元素. Defaults to 1.
            allow_repeat (bool, optional):
                允许重复值. Defaults to True.
            cmp (Callable[[_T, _T], Union[int, float]], optional):
                自定义比较符. 默认为比较符`-`.
        """
        self.__cmp = cmp
        self.__top_n = top_n
        self.__allow_repeat = allow_repeat
        self.__cache: List[_T] = []

    def step(self, value: Union[_T, List[_T]]) -> None:
        """更新值或值列表."""
        if isinstance(value, List):
            for v in value:
                self.__step_value(v)
        else:
            self.__step_value(value)

    @property
    def max_list(self) -> List[_T]:
        """返回top n列表."""
        return self.__cache

    @property
    def max(self) -> _T:
        """返回第top_n大的值."""
        return self.__cache[-1]

    def __step_value(self, value: _T) -> None:
        self.__cache.append(value)
        if not self.__allow_repeat:
            self.__cache = list(set(self.__cache))
        self.__cache.sort(key=cmp_to_key(self.__cmp),
                          reverse=True)  # type: ignore
        if len(self.__cache) > self.__top_n:
            self.__cache.pop()
