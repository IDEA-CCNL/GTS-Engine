"""提供参数集合(BaseArguments)管理功能的Mixin模块."""
from typing import List, Optional, Type, get_type_hints

from gts_common.framework import BaseArguments


class ArgsMixin:
    """提供参数集合(BaseArguments)管理功能的Mixin.

    Example:

    >>> # 定义该模块使用的参数集合类
    >>> class Arguments(BaseArguments):
    ...     ...
    ...
    >>> # 定义需要使用参数集合的类
    >>> class HasArgs(..., ArgsMixin, ...):
    ...
    ...     _args: Arguments  # 声明参数集合类型
    ...
    ...     def __init__(self, args_parse_list: Optional[List[str]] = None, ...):
    ...         ArgsMixin.__init__(self, args_parse_list)  # 实例化ArgsMixin
    ...         ...
    ...
    ...     def foo(self):
    ...         print(self._args)  # 参数在实例化时已经解析完成，在类方法中可以放心访问
    ...
    >>> has_args = HasArgs(args_parse_list=["--foo", "1"])  # 通过指定参数列表解析参数
    >>> has_args = HasArgs()  # 通过命令行获取参数
    >>> has_args._args
    """

    def __init__(self, args_parse_list: Optional[List[str]] = None):
        """实例化ArgsMixin.

        识别继承了ArgsMixin的子类声明的参数集合类，并在该类实例化时对参数进行解析。

        Args:
            args_parse_list (Optional[List[str]], optional):
                待解析的参数列表，为None时则从命令行获取。Defaults to None.

        Raises:
            Exception: 继承了ArgsMixin的类没有对应的参数集合类进行声明
        """
        # 识别参数集合类的声明
        try:
            arg_cls: Type[BaseArguments] = get_type_hints(type(self))["_args"]
        except Exception:
            raise Exception(
                "BaseTrainingPipeline should have property typing"
                " hint \"_args\" to indicate the BaseArguments class")
        # 解析参数
        arg_obj = arg_cls()
        self._args = arg_obj.parse_args(args_parse_list)
