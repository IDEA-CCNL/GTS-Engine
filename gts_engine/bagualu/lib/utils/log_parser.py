from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Optional, Type
import re
import os


Matcher = Callable[[str], Optional[str]]

#############################################################################################
## Base
#############################################################################################
class LogParser(metaclass=ABCMeta):
    
    def parse_log(self, file: str) -> Dict[str, Optional[str]]:
        if not os.path.exists(file):
            raise Exception("file not exist")
        log = self._load_log(file)
        return {name: self._support_error(matcher)(log) for name, matcher in self._target_info_dict.items()} 
    
    def _load_log(self, file: str) -> str:
        with open(file, "r", encoding="utf-8") as f:
            return f.read()
        
    @property
    @abstractmethod
    def _target_info_dict(self) -> Dict[str, Matcher]:
        return {}
    
    def _support_error(self, matcher: Matcher) -> Matcher:
        def wraper(log: str):
            try:
                res = matcher(log)
            except:
                res = None
            return res
        return wraper
    
    
    
#############################################################################################
## Derived
#############################################################################################
    
class StudentLogParser(LogParser):
    
    @property
    def _target_info_dict(self) -> Dict[str, Matcher]:
        return {
            "dev_acc": self.__dev_acc_matcher,
            "train_model_test_acc": self.__train_test_acc_matcher,
            "inf_model_test_acc": self.__inf_test_acc_matcher
        }

    def __dev_acc_matcher(self, log: str) -> Optional[str]:
        multi_model_dev_acc_list = re.findall(r"(?<=best checkpoint dev acc: )0\.\d+", log)
        if len(multi_model_dev_acc_list) > 0:
            return multi_model_dev_acc_list[0]
        else:
            return re.findall(r"(?<=save checkpoint. dev acc: )0\.\d+", log)[-1]
        
    def __train_test_acc_matcher(self, log: str) -> Optional[str]:
        return re.findall(r"(?<=INFO: training model test acc: )0\.\d+", log)[0]
    
    def __inf_test_acc_matcher(self, log: str) -> Optional[str]:
        return re.findall(r"(?<=INFO: inference model test acc: )0\.\d+", log)[0]

class TeacherLogParser(LogParser):
    
    @property
    def _target_info_dict(self) -> Dict[str, Matcher]:
        return {
            "test_acc": self.__test_acc_matcher,
        }
    
    def __test_acc_matcher(self, log: str) -> Optional[str]:
        return re.findall(r"(?<=global_acc: )0\.\d+", log)[0]


#############################################################################################
## Test
#############################################################################################

if __name__ == "__main__":
    file = "/raid/jiangyuzhen/Documents/gts_student_test/my_tests/unified_test/logs/tnews_student_2shots_1.log"
    file="/raid/jiangyuzhen/Documents/gts-student-finetune-std/out/student_output/logs/ft_std_csldcp_2022-10-31_16-50-19.log"
    res = StudentLogParser().parse_log(file)
    print(res)
    
        
    
    
