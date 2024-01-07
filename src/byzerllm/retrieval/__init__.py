import ray
from ray.types import ObjectRef
from byzerllm.records import ClusterSettings, EnvSettings, JVMSettings, TableSettings, SearchQuery, \
    ResourceRequirementSettings
from byzerllm.utils.retrieval import ByzerRetrieval, ClusterBuilder
from typing import List, Dict, Any, Optional, Union
from pyjava import PythonContext
from pyjava.udf import UDFBuilder
import importlib

DEFAULT_RETRIEVAL_UDF_NAME = "RetrievalVectorGateway"


class VectorEngine():
    Milvus = "milvus"


class VectorRetrieval(ByzerRetrieval):
    def __init__(self,
                 vector_engine: VectorEngine,
                 udf_name: str = DEFAULT_RETRIEVAL_UDF_NAME,
                 connection_args: Optional[dict[str, Any]] = None,
                 **kwargs
                 ):
        self.default_sys_conf = {
            "pythonMode": "ray",
            "maxConcurrency": 1
        }
        self.sys_conf = self.default_sys_conf.copy()
        self.verbose = kwargs.get("verbose", False)
        self.vector_engine = vector_engine

        self.connection_args = connection_args
        self.udf_name = udf_name if udf_name is not None and len(udf_name) > 0 else DEFAULT_RETRIEVAL_UDF_NAME
        self.retrieval_gateway = None

        self.context = PythonContext(
            0, [], self.sys_conf
        )
        self.setup("UDF_CLIENT", udf_name)
        self.context.have_fetched = True
        self.ray_context = self.context.rayContext

    def setup(self, name: str, value: Any) -> 'LemonRetrieval':
        self.sys_conf[name] = value
        self.context.conf = self.sys_conf
        return self

    def setup_num_workers(self, num_workers: int) -> 'LemonRetrieval':
        self.sys_conf["maxConcurrency"] = num_workers
        return self

    def launch_gateway(self) -> ray.actor.ActorHandle:
        try:
            self.retrieval_gateway = ray.get_actor(self.udf_name)
        except Exception:
            pass
        if self.retrieval_gateway:
            return self.retrieval_gateway
        retrieval_module = importlib.import_module(f'byzerllm.retrieval.{self.vector_engine}')

        def init_retrieval(_, conf: Dict[str, str]) -> Any:
            from byzerllm import consume_model
            consume_model(conf)
            retrieval = retrieval_module.Adapter(self.connection_args)
            return (retrieval, None)

        UDFBuilder.build(self.ray_context, init_retrieval, lambda _, v: None)
        return ray.get_actor(self.udf_name)

    def stop_gateway(self):
        try:
            gateway = ray.get_actor(self.udf_name)
            ray.kill(gateway)
        except ValueError:
            pass

    def is_gateway_exists(self) -> bool:
        try:
            ray.get_actor(self.udf_name)
            return True
        except ValueError:
            return False

    def gateway(self) -> ray.actor.ActorHandle:
        return ray.get_actor(self.udf_name)

    def get_table_settings(self, cluster_name: str, database: str, table: str) -> Optional[TableSettings]:
        return super().get_table_settings(cluster_name, database, table)

    def check_table_exists(self, cluster_name: str, database: str, table: str) -> bool:
        return super().check_table_exists(cluster_name, database, table)

    def create_table(self, cluster_name: str, tableSettings: TableSettings) -> bool:
        return super().create_table(cluster_name, tableSettings)

    def build(self, cluster_name: str, database: str, table: str, object_refs: List[ObjectRef[str]]) -> bool:
        return super().build(cluster_name, database, table, object_refs)

    def build_from_dicts(self, cluster_name: str, database: str, table: str, data: List[Dict[str, Any]]) -> bool:
        return super().build_from_dicts(cluster_name, database, table, data)

    def delete_by_ids(self, cluster_name: str, database: str, table: str, ids: List[Any]) -> bool:
        return super().delete_by_ids(cluster_name, database, table, ids)

    def get_tables(self, cluster_name: str) -> List[TableSettings]:
        return super().get_tables(cluster_name)

    def get_databases(self, cluster_name: str) -> List[str]:
        return super().get_databases(cluster_name)

    def shutdown_cluster(self, cluster_name: str) -> bool:
        return super().shutdown_cluster(cluster_name)

    def commit(self, cluster_name: str, database: str, table: str) -> bool:
        return super().commit(cluster_name, database, table)

    def truncate(self, cluster_name: str, database: str, table: str) -> bool:
        return super().truncate(cluster_name, database, table)

    def close(self, cluster_name: str, database: str, table: str) -> bool:
        return super().close(cluster_name, database, table)

    def closeAndDeleteFile(self, cluster_name: str, database: str, table: str) -> bool:
        return super().closeAndDeleteFile(cluster_name, database, table)

    def search_keyword(self, cluster_name: str, database: str, table: str, filters: Dict[str, Any], keyword: str,
                       fields: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        return super().search_keyword(cluster_name, database, table, filters, keyword, fields, limit)

    def search_vector(self, cluster_name: str, database: str, table: str, filters: Dict[str, Any], vector: List[float],
                      vector_field: str, limit: int = 10) -> List[Dict[str, Any]]:
        return super().search_vector(cluster_name, database, table, filters, vector, vector_field, limit)

    def search(self, cluster_name: str, search_query: Union[List[SearchQuery], SearchQuery]) -> List[Dict[str, Any]]:
        return super().search(cluster_name, search_query)

    def filter(self, cluster_name: str, search_query: Union[List[SearchQuery], SearchQuery]) -> List[Dict[str, Any]]:
        return super().filter(cluster_name, search_query)
