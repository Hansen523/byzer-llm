from typing import Optional, Any, Dict
from langchain_community.vectorstores.milvus import Milvus


class Adapter():
    def __init__(self, connection_args: Optional[dict[str, Any]] = None):
        self.milvus = Milvus(connection_args=connection_args)
