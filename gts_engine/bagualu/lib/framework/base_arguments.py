from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Optional, Dict, List, Type, Union, get_type_hints
from typing_extensions import Self
import sys

GeneralParser = Union[ArgumentParser, _ArgumentGroup]

class BaseArguments(Namespace, metaclass=ABCMeta):
    """支持自动解析嵌套结构的参数类
    
        通过继承，可以只需定义参数间的嵌套结构，实现形如
            main_arg
                - main_arg_param1
                - main_arg_param2
                - ...
                
                - sub_arg1
                    - sub_arg_param
                - sub_arg2
                    - ...
                - ...
        的嵌套参数，通过最外层参数的parse_args()方法自动完成解析

        > 使用方法参见 ./demo/arguments_base_demo.py
        
        > todo:
        > - [ ] 支持从类解析为命令行参数字符串
        > - [ ] 美化打印
    """
    
    @property
    def _sub_args_dict(self) -> Dict[str, Type["BaseArguments"]]:
        hintsDict = get_type_hints(type(self))
        return {key: val for key, val in hintsDict.items() if isinstance(val, type) and issubclass(val, BaseArguments)}
    
    @property
    def _arg_name(self) -> Optional[str]:
        return None
    
    @property
    def _arg_description(self) -> Optional[str]:
        return None
    
    def parse_args(self, parse_list: Optional[List[str]] = None) -> Self:
        # print(self._sub_args_dict)
        if len(sys.argv) > 1 and sys.argv[1] in {"--help", "-h"}:
            self._show_help()
        parser = ArgumentParser(prog=self._arg_name, description=self._arg_description)
        self._add_args(parser)
        _, left_params = parser.parse_known_args(parse_list, namespace=self)
        for arg_name, arg_cls in self._sub_args_dict.items():
            setattr(self, arg_name, arg_cls().parse_args(left_params))
        
        self._after_parse()
        return self
    
    def _add_args(self, parser: GeneralParser) -> None:
        pass
    
    def _show_help(self) -> None:
        parser = ArgumentParser(prog=self._arg_name, description=self._arg_description)
        self._add_all_args(parser)
        parser.parse_args()

    def _add_all_args(self, parser: ArgumentParser) -> ArgumentParser:
        parser_group = parser.add_argument_group(self._arg_name, self._arg_description)
        self._add_args(parser_group)
        for _, arg_cls in self._sub_args_dict.items():
            arg = arg_cls()
            arg._add_all_args(parser)
        return parser
    
    def _after_parse(self) -> None:
        pass
    
    
