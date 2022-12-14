from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import Protocol, Optional, List, Type, Tuple
import time
from dataclasses import dataclass
import weaviate
from alibabacloud_tea_util import models as util_models
from alibabacloud_ha3engine import models, client
import json



from .es_search_args import ESSearchArgs, SEARCH_CLIENT
from ...components import ProtocolArgs, LoggerManager
from ...framework.consts import RUN_MODE
from ...framework.mixin import OptionalLoggerMixin



#############################################################################################
## types
#############################################################################################

class ArgumentsProto(Protocol):
    train_data_path: Optional[str]
    unlabeld_data_path: Optional[str]
    start_time: float
    @property
    def run_mode(self) -> RUN_MODE: ...
    es_search_args: ESSearchArgs
    logger: Optional[str]
    
@dataclass
class SearchedSample:
    id: int
    content: str

Id_QueryVector = Tuple[int, List[float]]

#############################################################################################
## base
#############################################################################################


class SearchClient(OptionalLoggerMixin, metaclass=ABCMeta):
    #############################################################################################
    ##################################### public #############################################
    def __init__(self, args: ArgumentsProto) -> None:
        OptionalLoggerMixin.__init__(self, args.logger)
        self._args = args
        
    def es_search(self, id_query_vector: Id_QueryVector) -> Optional[List[SearchedSample]]:
        if self._already_run_time_min > self._args.es_search_args.es_max_runtime:
            self.info("es search exceeds maximum time")
            return
        content_list = self._fetch_data(id_query_vector[1])
        es_result = self._select_and_tokenize(id_query_vector[0], content_list)
        
        return es_result
    
    #############################################################################################
    ##################################### protected #############################################
    @property
    def _already_run_time_min(self) -> float:
        curr_time = time.time()
        already_run_time = curr_time -  self._args.es_search_args.start_time
        return already_run_time / 60
    
    def _select_and_tokenize(self, id: int, content_list: List[str]) -> List[SearchedSample]:
        content_list = sorted(content_list, key=lambda content : len(content), reverse=True)
        es_search_length = min(self._args.es_search_args.es_search_num - self._args.es_search_args.addtion_search, 
                               len(content_list))
        content_list = content_list[:es_search_length]
        return [SearchedSample(id, content) for content in content_list]
    
    #############################################################################################
    ##################################### virtual #############################################
    @abstractmethod
    def initializer(self) -> None:
        """?????????client????????????????????????
        """
        pass
    
    
    @abstractmethod
    def _fetch_data(self, query_vector: List[float]) -> List[str]:
        """????????????????????????

        Args:
            query_vector (List[float]): 

        Returns:
            List[str]: ???????????????????????????????????????
        """
        pass
    
#############################################################################################
## Derived
#############################################################################################
class WeaviateClient(SearchClient):
    
    @property
    def _server_url(self) -> str:
        if self._args.run_mode == RUN_MODE.OFFLINE:
            return ""  # ??????????????????
        elif self._args.run_mode == RUN_MODE.ONLINE:
            return ""  # ??????????????????
        else:
            raise Exception("mode is not supported")
    
    def __init__(self, args=None):
        super().__init__(args)
    
    def initializer(self) -> None:
        WeaviateClient.client = None
        try:
            WeaviateClient.client = weaviate.Client(self._server_url, timeout_config=(5, 90)) # type: ignore
        except TypeError:
            raise Exception('Error, can\'t connect to vector database server, is it running?')

    def _fetch_data(self, query_vector: List[float]) -> List[str]:
        # weavaite????????????
        table_name = "Generation"
        # ??????????????????
        # ?????????????????????
        vector_name = "vector"
        # ?????????????????????
        text_name = "content"
        # ?????????????????????????????????
        len_name = "counter"
        # ????????????
        # ????????????????????????
        nearVector = {vector_name: query_vector}
        # ???????????????????????????
        top = self._args.es_search_args.es_search_num
        # top = 10
        # ????????????????????????????????????????????????

        # with_where??????????????????????????????????????????????????????
        # operator?????????LessThan???GreaterThan???Equal
        if WeaviateClient.client:
            res = WeaviateClient.client.query.get(
                    table_name, [text_name]).with_near_vector(
                    nearVector).with_limit(
                    top).with_additional(
                    vector_name).do()
        else:
            raise Exception("client is not initialised")
        data = res['data']['Get']['Generation']
        es_result = ["".join(data_[text_name]) for data_ in data]
        return es_result

class WentianClient(SearchClient):
    def __init__(self, args=None):
        super().__init__(args)
        self.info('endpoint : {}'.format(self._args.es_search_args.endpoint))
        
    def initializer(self) -> None:
        endpoint = "" \
                if self._args.run_mode == RUN_MODE.OFFLINE else ""  # ??????????????????
        wentian_config = models.Config(
            endpoint=endpoint,
            instance_id="ha-cn-zvp2qr1sk01",
            protocol="http",
            access_user_name="ccnl",
            access_pass_word="2022idea"
        )
        # ???????????????????????????. ??????????????????????????????????????????. ?????? ms
        # ??????????????? push_documents_with_options ???????????????
        runtime = util_models.RuntimeOptions(
            connect_timeout=5000,
            read_timeout=10000,
            autoretry=False,
            ignore_ssl=False,
            max_idle_conns=50
        )
        
        # ????????? Ha3Engine Client
        WentianClient.ha3EngineClient = client.Client(wentian_config)


    def _fetch_data(self, query_vector: List[float]) -> List[str]:
        vector = ','.join(str(x) for x in query_vector)
        queryinfo="&&query=generation_base_index:\'{}&n={}\'".format(vector, str(self._args.es_search_args.es_search_num))
        query_str=f"kvpairs=first_formula:proxima_score(generation_base_index)&&sort=-RANK&&config=start:0,hit:{self._args.es_search_args.es_search_num},format:json&&cluster=general&&filter=fieldlen(content)>24"
        optionsHeaders = {}
        haSearchQuery = models.SearchQuery(query=query_str+queryinfo)
        haSearchRequestModel = models.SearchRequestModel(optionsHeaders, haSearchQuery)
        hastrSearchResponseModel = WentianClient.ha3EngineClient.search(haSearchRequestModel)
        data = json.loads(hastrSearchResponseModel.body.__str__()) # type: ignore
        content_items = data['result']['items']
        es_result = []
        for item in content_items:
            es_result.append(item['fields']['content'])
        return es_result


#############################################################################################
## factory
#############################################################################################
def search_client_factory(client: SEARCH_CLIENT) -> Type[SearchClient]:
    if client == SEARCH_CLIENT.WEAVIATE:
        return WeaviateClient
    elif client == SEARCH_CLIENT.WENTIAN:
        return WentianClient
    else:
        raise Exception("client is not supported")