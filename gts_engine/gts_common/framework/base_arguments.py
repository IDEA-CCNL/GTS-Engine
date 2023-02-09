"""参数集合基类模块."""
import sys
from abc import ABCMeta
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Dict, List, Optional, Type, Union, get_type_hints

from typing_extensions import Self

GeneralParser = Union[ArgumentParser, _ArgumentGroup]


class BaseArguments(Namespace, metaclass=ABCMeta):
    """参数集合基类.

    将参数定义、解析、类型标注进行整合的支持嵌套的用于模块化管理参数的参数集合基类

    Examples:

        * 参数定义（在argparser中定义参数）、参数声明一站式解决

        注：含如下ipython形式代码示例的docstring可以通过鼠标悬停在类名上在悬浮窗口中
        查看，ide会进行格式化和代码高亮。

        >>> class Arguments(BaseArguments):
        ...     foo: int  # 外部参数声明
        ...
        ...     # 外部参数定义
        ...     def _add_args(self, parser) -> None:
        ...         parser.add_argument("--foo", dest="foo", type=int)
        ...
        ...     bar: str = "123"  # 非外部传入参数定义
        ...
        ...     @property
        ...     def baz(self) -> int:  # 动态参数定义
        ...         return self.foo + 1
        ...
        >>> # 解析参数(不传入参数列表时则从命令行获取参数)
        >>> args = Arguments().parse_args(["--foo", "1"])
        >>> # 此时参数已被解析，且带有代码提示和类型推导（不声明、只定义就不会有代码提示）
        >>> args.foo
        1
        >>> args.bar
        "123"
        >>> args.baz
        2

        * 支持定义并自动解析嵌套结构的参数（父参数集合和子参数集合及子参数集合之间不能有传参flag冲突）
        * 支持打印help并以分组的方式显示子参数

        >>> # 定义三层嵌套参数参数，并添加名称和说明
        >>> class MainArguments(BaseArguments):
        ...
        ...     sub_args: "SubArguments"  # 声明子参数集合
        ...
        ...     foo: int
        ...
        ...     def _add_args(self, parser) -> None:
        ...         parser.add_argument("--foo", dest="foo", type=int)
        ...
        ...     @property
        ...     def _arg_name(self):
        ...         return "main arguments"
        ...
        ...     @property
        ...     def _arg_description(self):
        ...         return "description of main arguments"
        ...
        >>> class SubArguments(BaseArguments):
        ...
        ...     sub_sub_args: "SubSubArguments"
        ...
        ...     bar: int
        ...
        ...     def _add_args(self, parser) -> None:
        ...         parser.add_argument("--bar", dest="bar", type=int)
        ...
        ...     @property
        ...     def _arg_name(self):
        ...         return "sub arguments"
        ...
        ...     @property
        ...     def _arg_description(self):
        ...         return "description of sub arguments"
        ...
        >>> class SubSubArguments(BaseArguments):
        ...
        ...     baz: int
        ...
        ...     def _add_args(self, parser) -> None:
        ...         parser.add_argument("--baz", dest="baz", type=int)
        ...
        ...     @property
        ...     def _arg_name(self):
        ...         return "sub sub arguments"
        ...
        ...     @property
        ...     def _arg_description(self):
        ...         return "description of sub sub arguments"
        ...
        >>> # 只需调用最外层的解析，即可递归地解析所有子参数集合
        >>> args = MainArguments().parse_args(["--foo", "1", "--bar", "2", "--baz", "3"])
        >>> args.foo
        1
        >>> args.sub_args.bar
        2
        >>> args.sub_args.sub_sub_args.baz
        3
        >>> # 打印help
        >>> args = MainArguments().parse_args(["--help"])
        usage: main arguments [-h] [--foo FOO] [--bar BAR] [--baz BAZ]
        description of main arguments
        optional arguments:
        -h, --help  show this help message and exit
        main arguments:
        description of main arguments
        --foo FOO
        sub arguments:
        description of sub arguments
        --bar BAR
        sub sub arguments:
        description of sub sub arguments
        --baz BAZ

        * 支持定义后处理逻辑

        >>> # 定义路径参数，并在参数解析后创建路径
        >>> class Arguments(BaseArguments):
        ...
        ...     dir: Path
        ...
        ...     def _add_args(self, parser) -> None:
        ...         # type参数支持任何可以通过str实例化的类
        ...         parser.add_argument("--dir", dest="dir", type=Path)
        ...
        ...     def _after_parse(self) -> None:
        ...         if not self.dir.exists():
        ...             print(f"make directory {self.dir}")
        ...             self.dir.mkdir()
        ...
        >>> args = Arguments().parse_args(["--dir", "/raid/jiangyuzhen/Documents/gts-engine/tmp_dir"])
        make directory /raid/jiangyuzhen/Documents/gts-engine/tmp_dir
        >>> args.dir
        PosixPath('/raid/jiangyuzhen/Documents/gts-engine/tmp_dir')

    Todo:
        - [ ] (Jiang Yuzhen) 美化参数集合打印输出格式
        - [ ] (Jiang Yuzhen) 尝试把参数定义合并到声明中，例如仿照`dataclass`的`field`函数，通过在类中声明
            `foo: int = field(flag="--foo", default=1, help="description")`
            自动执行`parser.add_argument("--foo", default=1, type=int, help="description")`
        - [ ] (Jiang Yuzhen) 尝试将参数管理全局化（不一定通过修改本模块来实现）
    """

    # ========================== public ===============================

    def parse_args(self, parse_list: Optional[List[str]] = None) -> Self:
        """递归地解析参数集合及其子参数集合.

        Args:
            parse_list (Optional[List[str]], optional):
                参数列表，为None时则从命令行解析参数。Defaults to None.

        Returns:
            Self: 解析后的参数对象
        """
        # 支持`--help`参数
        if ((len(sys.argv) > 1 and sys.argv[1] in {"--help", "-h"})
                or (parse_list is not None and len(parse_list) == 1
                    and parse_list[0] in {"--help", "-h"})):
            self._show_help()  # 程序会在这里打印help并结束
        parser = ArgumentParser(prog=self._arg_name,
                                description=self._arg_description)
        # 加入并解析当前层参数
        self._add_args(parser)
        _, left_params = parser.parse_known_args(parse_list, namespace=self)
        # 将剩余参数传入子参数集合
        for arg_name, arg_cls in self._sub_args_dict.items():
            setattr(self, arg_name, arg_cls().parse_args(left_params))
        # 调用后处理逻辑
        self._after_parse()
        return self

    # ========================== abstract ===============================

    def _add_args(self, parser: GeneralParser) -> None:
        """将参数加入parser."""
        pass

    def _after_parse(self) -> None:
        """后处理逻辑，可用于需要传参结束后进行处理和计算的逻辑."""
        pass

    @property
    def _arg_name(self) -> Optional[str]:
        """参数集合名称，用于help."""
        return None

    @property
    def _arg_description(self) -> Optional[str]:
        """参数集合描述，用于help."""
        return None

    # ========================== protected ===============================

    @property
    def _sub_args_dict(self) -> Dict[str, Type["BaseArguments"]]:
        """子参数集合类字典."""
        hintsDict = get_type_hints(type(self))
        return {
            key: val
            for key, val in hintsDict.items()
            if isinstance(val, type) and issubclass(val, BaseArguments)
        }

    def _show_help(self) -> None:
        """打印help."""
        parser = ArgumentParser(prog=self._arg_name,
                                description=self._arg_description)
        self._add_all_args(parser)
        parser.parse_args(["--help"])

    def _add_all_args(self, parser: ArgumentParser) -> ArgumentParser:
        """为了打印help，将子参数集合的参数都加入本层parser."""
        parser_group = parser.add_argument_group(self._arg_name,
                                                 self._arg_description)
        self._add_args(parser_group)
        for _, arg_cls in self._sub_args_dict.items():
            arg = arg_cls()
            arg._add_all_args(parser)
        return parser
