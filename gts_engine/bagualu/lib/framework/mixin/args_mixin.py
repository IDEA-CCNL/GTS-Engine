from typing import Type, Optional, List, get_type_hints

from ..base_arguments import BaseArguments

class ArgsMixin:
    
    def __init__(self, args_parse_list: Optional[List[str]] = None):
        try:
            arg_cls: Type[BaseArguments] = get_type_hints(type(self))["_args"]
        except:
            raise Exception("BaseTrainingPipeline should have property typing hint \"_args\" to indicate the BaseArguments class")
        arg_obj = arg_cls()
        self._args = arg_obj.parse_args(args_parse_list)