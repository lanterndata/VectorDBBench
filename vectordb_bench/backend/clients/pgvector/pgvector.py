"""Wrapper around the Pgvector vector database over VectorDB"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Tuple, Type
from functools import wraps

from ..api import VectorDB, DBConfig, DBCaseConfig, IndexType
from pgvector.sqlalchemy import Vector
from .config import PgVectorConfig, _pgvector_case_config
from sqlalchemy import (
    MetaData,
    create_engine,
    insert,
    select,
    Index,
    Table,
    text,
    Column,
    Float, 
    Integer
)
from sqlalchemy.orm import (
    declarative_base, 
    mapped_column, 
    Session
)

log = logging.getLogger(__name__) 

class PgVector(VectorDB):
    """ Use SQLAlchemy instructions"""
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "PgVectorCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self._index_name = "pqvector_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        # construct basic units
        pg_engine = create_engine(**self.db_config)
        Base = declarative_base()
        pq_metadata = Base.metadata
        pq_metadata.reflect(pg_engine) 
        
        # create vector extension
        with pg_engine.connect() as conn: 
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            conn.commit()
        
        self.pg_table = self._get_table_schema(pq_metadata)
        if drop_old and self.table_name in pq_metadata.tables:
            log.info(f"Pgvector client drop table : {self.table_name}")
            # self.pg_table.drop(pg_engine, checkfirst=True)
            pq_metadata.drop_all(pg_engine)
            self._create_table(dim, pg_engine)

    
    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return PgVectorConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return _pgvector_case_config.get(index_type)

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.pg_engine = create_engine(**self.db_config)

        Base = declarative_base()
        pq_metadata = Base.metadata
        pq_metadata.reflect(self.pg_engine) 
        self.pg_session = Session(self.pg_engine)
        self.pg_table = self._get_table_schema(pq_metadata)
        yield 
        self.pg_session = None
        self.pg_engine = None 
        del (self.pg_session)
        del (self.pg_engine)
    
    def ready_to_load(self):
        pass

    def optimize(self):
        pass

    def ready_to_search(self):
        pass

    def _get_table_schema(self, pq_metadata):
        return Table(
            self.table_name,
            pq_metadata,
            Column(self._primary_field, Integer, primary_key=True),
            Column(self._vector_field, Vector(self.dim)),
            extend_existing=True
        )
    
    def _create_index(self, pg_engine):
        index_param = self.case_config.index_param()
        if self.case_config.index == IndexType.IVFFlat:
            index = Index(self._index_name, self.pg_table.c.embedding,
                postgresql_using='ivfflat',
                postgresql_with={'lists': index_param["lists"]},
                postgresql_ops={'embedding': index_param["metric"]}
            )
        else:
            index = Index(self._index_name, self.pg_table.c.embedding,
                postgresql_using='hnsw',
                postgresql_with={'m': index_param["m"], 'ef_construction': index_param['ef_construction']},
                postgresql_ops={'embedding': index_param["metric"]}
            )
            
        index.drop(pg_engine, checkfirst = True)
        index.create(pg_engine)

    def _create_table(self, dim, pg_engine : int):
        try:
            # create table
            self.pg_table.create(bind = pg_engine, checkfirst = True)
            # create vec index
            self._create_index(pg_engine)
        except Exception as e:
            log.warning(f"Failed to create pgvector table: {self.table_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        try:
            items = [dict(id = metadata[i], embedding=embeddings[i]) for i in range(len(metadata))]
            self.pg_session.execute(insert(self.pg_table), items)
            self.pg_session.commit()
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
    ) -> Tuple[list[int], float]:
        assert self.pg_table is not None
        search_param = self.case_config.search_param()
        filter_statement = ''
        vec_id = None
        if filters:
            vec_id = filters.get('id')
            filter_statement = f'WHERE "{self._primary_field}" > {vec_id}'
        
        operator_str = search_param['metric_op']

        if self.case_config.index == IndexType.IVFFlat:
            self.pg_session.execute(text(f'SET ivfflat.probes = {search_param["probes"]}'))
        else:
            self.pg_session.execute(text(f'SET hnsw.ef_search = {search_param["ef"]}'))
 
        statement = text(f'SELECT "{self._primary_field}" FROM "{self.pg_table}" {filter_statement} ORDER BY "{self._vector_field}" {operator_str} \'{query}\' LIMIT {k}')
        s = time.perf_counter()
        res = self.pg_session.execute(statement)
        dur = time.perf_counter() - s
        return [row[0] for row in res.fetchall()], dur
