"""Wrapper around the TimescaleDB vector database over VectorDB"""
from io import StringIO
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Tuple, Type
from datetime import timedelta
from timescale_vector import client
import psycopg2

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
        pg_engine = psycopg2.connect(self.db_config['url'])
        pg_engine.autocommit = True
        self.pg_session = pg_engine.cursor()

        yield 
        self.client = None
        self.pg_session = None
        del (self.client)
        del (self.pg_session)
    
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
            f = StringIO("")
            for i in range(len(metadata)):
                f.write(f"{uuid.uuid1()}\t{{\"id\": {metadata[i]}}}\t{embeddings[i]}")
                if i != len(metadata) - 1:
                    f.write('\n')
            f.seek(0)
            self.pg_session.copy_expert(f'COPY "{self.table_name}" ({self._primary_field}, metadata, {self._vector_field}) FROM STDIN', f)
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
            self.pg_session.execute(f'SELECT "{self._primary_field}" FROM "{self.table_name}" ORDER BY "{self._vector_field}" <=> {query} LIMIT {k}')
            res = self.pg_session.fetchall()
            return [row[0] for row in res]

        return [row[1]['id'] for row in records] 
