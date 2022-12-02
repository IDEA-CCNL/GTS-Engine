import json
from typing import Any, Callable, List, Dict, Type, TypeVar, Union, Generator, Union, Iterable
from dataclasses import asdict
from pathlib import Path
from pydantic import BaseModel
from pydantic.json import pydantic_encoder

def load_json(file: Union[str, Path]) -> Union[List, Dict]:
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(obj: Union[Dict, List], file: Union[str, Path], indent: int = 0) -> None:
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)

def dump_json_list(obj: List, file: Union[str, Path]) -> None:
    with open(file, 'w', encoding='utf-8') as f:
        f.writelines([json.dumps(content, ensure_ascii=False) + "\n" for content in obj])
        

def load_json_list(file: Union[str, Path]) -> Generator[Union[Dict, List], None, None]:
    with open(file, 'r', encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def dump_dataclass_json_list(dataclass_list: List, file: Union[str, Path]) -> None:
    json_list = [asdict(dc) for dc in dataclass_list]
    dump_json_list(json_list, file)

T = TypeVar("T")
def load_dataclass_json_list(file: Union[str, Path], dataclass: Type[T]) -> List[T]:
    json_list = load_json_list(file)
    dataclass_list =  [dataclass(**d) for d in json_list] # type: ignore
    return dataclass_list

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)
def load_pydantic_model_json_list(file: Union[str, Path], pydantic_model: Type[PydanticModel]) -> Iterable[PydanticModel]:
    for dict_ in load_json_list(file):
        yield pydantic_model(**dict_) # type: ignore
    
def dump_pydantic_json(obj: Any, file: Union[str, Path], indent: int = 0, sort_keys: bool = False) -> None:
    with open(file, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent, ensure_ascii=False, default=pydantic_encoder)

if __name__ == "__main__":
    file = "/raid/jiangyuzhen/Documents/students_datasets/csldcp/train_0.json"
    line_iter = load_json_list(file)
    line = next(line_iter)
    print(line)