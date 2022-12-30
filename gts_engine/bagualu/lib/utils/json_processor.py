import json
from dataclasses import asdict
from pathlib import Path
from pydantic import BaseModel, parse_obj_as
from pydantic.json import pydantic_encoder
from typing import List, Union, TypeVar, Type, Any, Optional, Tuple, Dict, Iterable, overload

LoadType = TypeVar("LoadType", bound=Type)

@overload
def load_json(file: Union[str, Path]) -> Any: ...

@overload
def load_json(file: Union[str, Path], type_: Type[LoadType]) -> LoadType: ...

@overload
def load_json_list(file: Union[str, Path]) -> Iterable[Any]: ...

@overload
def load_json_list(file: Union[str, Path], type_: Type[LoadType]) -> Iterable[LoadType]: ...

def dump_json(obj: Any, file: Union[str, Path], indent: Optional[int] = None) -> None:
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=pydantic_encoder)

def dump_json_list(obj: List[Any], file: Union[str, Path]) -> None:
    with open(file, 'w', encoding='utf-8') as f:
        f.writelines([json.dumps(content, ensure_ascii=False, default=pydantic_encoder) + "\n" for content in obj])

def load_json(file: Union[str, Path], type_: Optional[Type[LoadType]] = None) -> Union[Any, LoadType]:
    with open(file, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if type_ is None:
        return obj
    else:
        return parse_obj_as(type_, obj)

def load_json_list(file: Union[str, Path], type_: Optional[Type[LoadType]] = None) -> Union[Iterable[Any], Iterable[LoadType]]:
    with open(file, 'r', encoding="utf-8") as f:
        for line in f:
            if type_ is None:
                yield json.loads(line)
            else:
                yield parse_obj_as(type_, json.loads(line))

