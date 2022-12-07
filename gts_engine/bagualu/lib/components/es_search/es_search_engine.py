from dataclasses import dataclass, asdict
import os
from random import shuffle
from typing import Callable, Dict, List, Optional, NamedTuple, Protocol, Generator, Tuple
from copy import copy
from sentence_transformers import SentenceTransformer
import torch
import time
import multiprocessing as mp
import json
from tqdm import tqdm

from .es_search_args import ESSearchArgs
from .es_search_client import search_client_factory
from ...framework.classification_finetune.consts import Label2Token, LabeledSample
from ...framework.classification_finetune import DataReaderClf
from ...framework.consts import RUN_MODE
from ...components import ProtocolArgs, LoggerManager
from ...framework.mixin import OptionalLoggerMixin


os.environ["TOKENIZERS_PARALLELISM"] = "false"


#############################################################################################
## types
#############################################################################################

class ArgumentsProto(Protocol):
    train_data_path: Optional[str]
    unlabeld_data_path: Optional[str]
    es_search_args: ESSearchArgs
    start_time: float
    run_mode: RUN_MODE
    logger: Optional[str]
    gpu_id: int
    pretrained_model_root: str

Id_Text = Tuple[int, str]
Id_QueryVector = Tuple[int, List[float]]

#############################################################################################
## engine
#############################################################################################

class ESSearchEngine(OptionalLoggerMixin):
    """es检索器
    """
    
    #############################################################################################
    ######################################## public ##########################################
    
    def __init__(
        self, 
        args: ArgumentsProto, 
        label2token: Label2Token
    ):
        OptionalLoggerMixin.__init__(self, args.logger)
        self._args = args
        self._label2token = label2token
    
    def es_search_augment(self) -> None:
        # 执行条件
        if self._args.es_search_args.augm_es_path is None:
            raise Exception("path for es search file is not passed")
        if os.path.exists(self._args.es_search_args.augm_es_path):
            self.info("es file already exists, skip es search")
            return
        # 加载和计算数据
        text_dict = self.__load_text_dict()
        search_text_list = self.__gen_search_text_list(text_dict)
        self._args.es_search_args.es_search_num = self.__cal_es_search_num(search_text_list)
        if len(search_text_list) == 0 or self._args.es_search_args.es_search_num == 0:
            self.info("no text needs to be augmented, skip es search")
            return
        self.info(f"length of the text list for es search(`len(search_text_list)`): {len(search_text_list)}")
        self.info(f"search number for each text(`es_search_args.es_search_num`)：{self._args.es_search_args.es_search_num}")
        # 生成搜索向量
        cuda_id = f"cuda:{self._args.gpu_id}"
        device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")
        model = self.__load_sbert_client(device)
        query_vector_list = self.__gen_query_vector_list(search_text_list, model, device)
        self.info(f"query vectors num: {len(query_vector_list)}")
        # 启动搜索
        self._args.es_search_args.start_time = time.time()
        client = search_client_factory(self._args.es_search_args.es_cllient)(self._args)
        es_search_result_num = 0
        with open(self._args.es_search_args.augm_es_path, 'w', encoding='utf8') as output_file, \
                mp.Pool(processes=self._args.es_search_args.es_num_thread, initializer=client.initializer) as pool:
            iters = pool.imap(client.es_search, query_vector_list)
            for searched_sample_list in tqdm(iters, desc='es Augmentation'):
                if searched_sample_list is not None:
                    output_file.writelines([json.dumps(asdict(searched_sample), ensure_ascii=False) + "\n" 
                                            for searched_sample in searched_sample_list])
                    es_search_result_num += len(searched_sample_list)
        es_end_time = time.time()
        # 输出信息
        self.info('the spend time of es search :%d' % (es_end_time - self._args.es_search_args.start_time))
        self.info('es search num:%d' % (es_search_result_num))
        del model
            
    #############################################################################################
    ######################################## private ##########################################
            
    def __load_text_dict(self) -> Dict[str, List[Id_Text]]:
        """按label加载数据字典
        {
            "label1": [(id1, text1), (id2, text2), ...]
            "label2": ...
            ...
            "unlabeled": ...
        }
        """
        text_dict: Dict[str, List[Id_Text]] = {}
        if self._args.train_data_path is not None and os.path.exists(self._args.train_data_path):
            if self._args.es_search_args.add_label_description:
                sample_handler: Callable[[LabeledSample], Id_Text] = lambda sample : (sample.id, sample.text + ',这是' + sample.label)
            else:
                sample_handler: Callable[[LabeledSample], Id_Text] = lambda sample : (sample.id, sample.text)
            for sample in DataReaderClf.read_labeled_sample(self._args.train_data_path, self._label2token):
                if sample.label not in text_dict.keys():
                    text_dict[sample.label] = []
                text_dict[sample.label].append(sample_handler(sample))
        if self._args.unlabeld_data_path is not None and os.path.exists(self._args.unlabeld_data_path):
            text_dict["unlabeled"] = [(sample.id, sample.text) for sample in DataReaderClf.read_unlabeled_sample(self._args.unlabeld_data_path)]
            
        return text_dict
    
    def __gen_search_text_list(self, text_dict: Dict[str, List[Id_Text]]) -> List[Id_Text]:
        """根据条件加载需要进行搜索的句子列表
        """
        text_num = sum([len(val) for val in text_dict.values()])
        search_text_list: List[Id_Text] = []
        # 如果训练文本数量足够，则无需增强
        if text_num > self._args.es_search_args.max_augment_data or text_num == 0:
            pass
        # 如果训练文本数量不足但超出搜索query上限，需要筛选用于搜索的text
        elif self._args.es_search_args.stratification_sample and text_num > self._args.es_search_args.es_search_query:
            labeled_text_dict = copy(text_dict)
            unlabeled_list = []
            if "unlabeled" in labeled_text_dict.keys():
                unlabeled_list = labeled_text_dict.pop("unlabeled")
            label_num = len(labeled_text_dict)
            search_num_each_label = int(self._args.es_search_args.es_search_query / label_num) \
                                if label_num > 0 else 0
            # 优先使用标注数据
            for text_list in labeled_text_dict.values():
                shuffle(text_list)
                search_text_list += text_list[:search_num_each_label]
            # 多余的query用无标注数据填充
            unlabeled_search_num = self._args.es_search_args.es_search_query - len(search_text_list)
            if unlabeled_search_num > 0:
                search_text_list += unlabeled_list[:unlabeled_search_num]
        # 否则使用所有的text进行搜索
        else:
            for text_list in text_dict.values():
                search_text_list += text_list
        
        return search_text_list
    
    def __cal_es_search_num(self, search_text_list: List[Id_Text]) -> int:
        """计算每条text需要搜索的结果数量"""
        es_search_num = int(self._args.es_search_args.max_augment_data / len(search_text_list)) + \
                        self._args.es_search_args.addtion_search \
                    if len(search_text_list) > 0 else 0
        return min(4000, es_search_num) # 每条数据搜索量不超过4k
     
    def __load_sbert_client(self, device) -> SentenceTransformer:
        """加载sbert"""
        self.info("loading sbert for es search...")
        if self._args.run_mode == RUN_MODE.OFFLINE:
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device) 
        else:
            model_root = self._args.pretrained_model_root
            model_path = os.path.join(model_root, 'paraphrase-multilingual-MiniLM-L12-v2')
            self.info(f"sentence transformer path: {model_path}")
            model = SentenceTransformer(model_path, device=device)
        self.info("loading sbert for es search...finished")
        return model
    
    def __gen_query_vector_list(self, search_text_list: List[Id_Text], model: SentenceTransformer, device) -> List[Id_QueryVector]:
        """计算查询向量"""
        id_query_vector_list: List[Tuple[int, torch.Tensor]] = []
        batch_size = self._args.es_search_args.search_batch_size
        for idx in range(0, len(search_text_list), batch_size):
            id_text_batch = search_text_list[idx : idx + batch_size]
            text_batch = [id_text[1] for id_text in id_text_batch]
            query_vector_batch: List[torch.Tensor] = model.encode(text_batch, device=device) # type: ignore
            id_query_vector_list += [(id_text_batch[i][0], query_vector_batch[i]) for i in range(len(id_text_batch))]
        return [(id_query_vector[0], id_query_vector[1].tolist()) for id_query_vector in id_query_vector_list]
        