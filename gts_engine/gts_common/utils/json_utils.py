"""json读写工具集.

提供支持类别推导、校验，支持json和json列表（每行为一个json字符串）形式的读写工具

Todo:
    - [ ] (Jiang Yuzhen) 修改模块名为json_utils或更合适的其他名称
"""
import json
from pathlib import Path
from typing import (Any, Generator, List, Optional, Type, TypeVar, Union,
                    overload)

from pydantic import parse_obj_as
from pydantic.json import pydantic_encoder

LoadType = TypeVar("LoadType", bound=Type)


@overload
def load_json(file: Union[str, Path]) -> Any:
    """读取json文件.

    Args:
        file (Union[str, Path]): 文件路径

    Returns:
        Any: json文件对应的对象
    """


@overload
def load_json(file: Union[str, Path], type_: Type[LoadType]) -> LoadType:
    """读取指定类型的json文件.

    Args:
        file (Union[str, Path]):
            json文件路径
        type_ (Type[LoadType]):
            json文件对应的对象类型，支持List、Dict等基本数据类型与
            dataclass、pydantic.BaseModel等结构体的任意嵌套

    Returns:
        LoadType: json文件对象，并检查、解析为 type_ 参数指定类型

    Todo:
        - [ ] (Jiang Yuzhen) 将文件类型解析错误捕获抛出，并给出文件信息

    Example:
        json文件如下
        -----------------------------------
        [
            {
                "id": 1,
                "name": "Sam"
            },
            {
                "id": 2,
                "name": "Lucy"
            }
        ]
        -----------------------------------

        >>> # 定义结构体，dataclass、pydantic.BaseModel均可支持
        >>> @dataclass
        ... class User:
        ...    id: int
        ...    name: str
        ...
        >>> user_list = load_json(file=json_path, type_=List[User])
        >>> user_list
        [User(id=1, name='Sam'), User(id=2, name='Lucy')]

        此时user_list被加载为List[User]，且类型推导也为List[User]
    """


@overload
def load_json_list(file: Union[str, Path]) -> Generator[Any, None, None]:
    """读取json列表文件（每行为一个json对象）

    Args:
        file (Union[str, Path]): json文件路径

    Yields:
        Any: 依次生成每一行的json对象
    """


@overload
def load_json_list(file: Union[str, Path],
                   type_: Type[LoadType]) -> Generator[LoadType, None, None]:
    """读取json列表文件（每行为一个json对象），并检查、解析为指定类型.

    Args:
        file (Union[str, Path]): json文件路径
        type_ (Type[LoadType]): 列表元素（每一行的json对象）对象类型

    Yields:
        LoadType: 依次生成每一行的json对象，并检查、解析为 type_ 参数指定的类型

    Todo:
        - [ ] (Jiang Yuzhen) 将文件类型解析错误捕获抛出，并定位到类型错误的行/json字符串

    Example:
        json文件如下
        -----------------------------------
        {"id": 1, "name": "Sam"}
        {"id": 2, "name": "Lucy"}
        -----------------------------------

        >>> # 定义结构体，dataclass、pydantic.BaseModel均可支持
        >>> @dataclass
        ... class User:
        ...     id: int
        ...     name: str
        ...
        >>> for user in load_json_list(file=json_path, type_=User):
        ...     user
        ...
        >>> user_list = list(load_json_list(file=json_path, type_=User))
        >>> user_list
        [User(id=1, name='Sam'), User(id=2, name='Lucy')]

        此时user_list被加载为List[User]，且类型推导也为List[User]
    """


def dump_json(obj: Any,
              file: Union[str, Path],
              indent: Optional[int] = None) -> None:
    """将对象写入json文件.

    Args:
        obj (Any):
            解析为json的对象，支持List、Dict等基本数据类型与
            dataclass、pydantic.BaseModel等结构体的任意嵌套
        file (Union[str, Path]):
            写入文件路径
        indent (Optional[int], optional):
            json换行缩进空格数. Defaults to None.
    """
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(obj,
                  f,
                  indent=indent,
                  ensure_ascii=False,
                  default=pydantic_encoder)


def dump_json_list(obj: List[Any], file: Union[str, Path]) -> None:
    """将对象列表按行写入json文件，每一行为一个对象对应的json字符串.

    Args:
        obj (List[Any]): 写入的对象列表
        file (Union[str, Path]): json文件路径
    """
    with open(file, 'w', encoding='utf-8') as f:
        f.writelines([
            json.dumps(content, ensure_ascii=False, default=pydantic_encoder) +
            "\n" for content in obj
        ])


def load_json(file: Union[str, Path],
              type_: Optional[Type[LoadType]] = None) -> Union[Any, LoadType]:
    """load_json()函数实现."""
    with open(file, encoding="utf-8") as f:
        obj = json.load(f)
    if type_ is None:
        return obj
    else:
        return parse_obj_as(type_, obj)


def load_json_list(
    file: Union[str, Path],
    type_: Optional[Type[LoadType]] = None
) -> Union[Generator[Any, None, None], Generator[LoadType, None, None]]:
    """load_json_list() 函数实现."""
    with open(file, encoding="utf-8") as f:
        for line in f:
            if type_ is None:
                yield json.loads(line)
            else:
                yield parse_obj_as(type_, json.loads(line))


if __name__ == "__main__":
    from dataclasses import dataclass

    json_path = "/raid/jiangyuzhen/Documents/gts-engine/ig_test.json"
    """
    {"id": 1, "name": "Sam"}
    {"id": 2, "name": "Lucy"}
    """

    @dataclass
    class User:
        id: int
        name: str

    user_list = list(load_json_list(file=json_path, type_=User))
    print(user_list)
    dump_json_list(obj=user_list, file=json_path)
