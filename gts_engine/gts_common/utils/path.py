"""路径相关工具集.

Todo:
    - [ ] (Jiang Yuzhen) 将模块名改为path_utils或类似名称，和json_utils等工具集保持一致
"""
import os
import shutil
from pathlib import Path
from typing import Literal, Union


def mk_inexist_dir(path: Union[str, Path],
                   r: bool = True,
                   clean: bool = False):
    """创建不存在的路径.

    Args:
        path: 路径
        r: 是否递归地创建父路径. Defaults to True.
        clean: 如果路径不为空，是否清空
    """
    if not os.path.exists(path):
        if r:
            os.makedirs(path)
        else:
            os.mkdir(path)
    if clean:
        shutil.rmtree(path)
        os.mkdir(path)


def get_file_size(path: Union[str, Path], unit: Literal["kb", "mb",
                                                        "gb"]) -> float:
    """获取文件在指定单位下的大小.

    Todo:
        - [ ] (Jiang Yuzhen) 预先检查该路径是否为文件，否则抛出错误
    """
    if unit == "kb":
        scale = 1024
    elif unit == "mb":
        scale = 1024**2
    elif unit == "gb":
        scale = 1024**3
    else:
        raise Exception("unit is not supported")
    size = os.path.getsize(path)
    return round(size / scale, 4)
