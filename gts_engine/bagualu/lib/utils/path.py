import os
from typing import Literal
import shutil
from typing import Union
from pathlib import Path

def mk_inexist_dir(path: Union[str, Path], r: bool = True, clean: bool = False):
    """创建不存在的路径 

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
        
def get_file_size(path: Union[str, Path], unit: Literal["kb", "mb", "gb"]) -> float:
    """获取文件在指定单位下的大小"""
    if unit == "kb":
        scale = 1024
    elif unit == "mb":
        scale = 1024 ** 2
    elif unit == "gb":
        scale = 1024 ** 3
    else:
        raise Exception("unit is not supported")
    size = os.path.getsize(path)
    return round(size / scale, 4)
    
