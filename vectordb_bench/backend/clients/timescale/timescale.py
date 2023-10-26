"""Wrapper around the TimescaleDB vector database over VectorDB"""
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Tuple, Type
from datetime import timedelta
from timescale_vector import client

from ..api import EmptyDBCaseConfig, VectorDB, DBConfig, DBCaseConfig, IndexType
from .config import TimescaleConfig
log = logging.getLogger(__name__) 

class Timescale(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "TimescaleCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self._index_name = "timescale_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        # construct basic units
        ts_client = client.Sync(db_config['url'], 
                   self.table_name,  
                   self.dim, 
                   time_partition_interval=timedelta(days=7))
        
        if drop_old:
            try:
                ts_client.drop_table()
            except:
                pass
            ts_client.create_tables()

    
    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return TimescaleConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return EmptyDBCaseConfig

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.client = client.Sync(self.db_config['url'], 
                   self.table_name,  
                   self.dim, 
                   time_partition_interval=timedelta(days=7))

        yield 
        self.client = None
        del (self.client)
    
    def ready_to_load(self):
        pass

    def optimize(self):
        self._create_index(self.client)

    def ready_to_search(self):
        pass

    def _create_index(self, ts_client):
        ts_client.create_embedding_index(client.TimescaleVectorIndex())

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> Tuple[int, Exception|None]:
        try:
            items = [(uuid.uuid1(), {"id": metadata[i]}, "", embeddings[i]) for i in range(len(metadata))]
            self.client.upsert(items)
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into pgvector table ({self.table_name}), error: {e}")   
            return 0, e

    def search_embedding(        
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        if filters:
            records = self.client.search(query, limit=k, filter={"id": filters.get('id')})
        else:
            records = self.client.search(query, limit=k)

        return [row[1]['id'] for row in records] 
