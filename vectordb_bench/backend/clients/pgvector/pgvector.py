"""Wrapper around the Pgvector vector database over VectorDB"""

from io import StringIO, BytesIO
import psycopg2
import logging
from contextlib import contextmanager
from typing import Any, Tuple, Type

from ..api import PgDB, DBConfig, DBCaseConfig, IndexType
from .config import PgVectorConfig, _pgvector_case_config

log = logging.getLogger(__name__) 

class PgVector(PgDB):
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
        self.f = StringIO("")
        self.binary_f = BytesIO(b"")

        self._index_name = "pgvector_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        # construct basic units
        pg_engine = psycopg2.connect(db_config['url'])
        pg_engine.autocommit = True
        pg_session = pg_engine.cursor()
        
        # create vector extension
        pg_session.execute('CREATE EXTENSION IF NOT EXISTS vector')
        if drop_old:
            log.info(f"Pgvector client drop table : {self.table_name}")
            pg_session.execute(f'DROP TABLE IF EXISTS "{self.table_name}" CASCADE')
            self._create_table(pg_session, dim)

    
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
        self.pg_engine = psycopg2.connect(self.db_config['url'])
        self.pg_engine.autocommit = True
        self.pg_session = self.pg_engine.cursor()
        search_param = self.case_config.search_param()
        
        if self.case_config.index == IndexType.IVFFlat:
            self.pg_session.execute(f'SET ivfflat.probes = {search_param["probes"]}')
        else:
            self.pg_session.execute(f'SET hnsw.ef_search = {search_param["ef"]}')
            
        yield 
        self.pg_session = None
        self.pg_engine = None 
        del (self.pg_session)
        del (self.pg_engine)
    
    def ready_to_load(self):
        pass

    def optimize(self):
        # create vec index
        self._create_index(self.pg_session)

        self.pg_session.execute("SELECT 1 FROM pg_extension WHERE extname='pg_prewarm'")
        res = self.pg_session.fetchall()

        if len(res) != 0:
            self.pg_session.execute(f"SELECT pg_prewarm('{self._index_name}')")

    def ready_to_search(self):
        pass
    
    def _create_index(self, pg_session):
        index_param = self.case_config.index_param()
        if self.case_config.index == IndexType.IVFFlat:
            pg_session.execute(f'CREATE INDEX "{self._index_name}" ON "{self.table_name}" USING ivfflat("{self._vector_field}" {index_param["metric"]}) WITH (lists={index_param["lists"]})')
        else:
            pg_session.execute(f'''
              SET hnsw.external_index_host='{index_param["external_index_host"]}';
              SET hnsw.external_index_port={index_param["external_index_port"]};
              SET hnsw.external_index_secure={index_param["external_index_secure"]};
            ''')
            pg_session.execute(f'CREATE INDEX "{self._index_name}" ON "{self.table_name}" USING hnsw("{self._vector_field}" {index_param["metric"]}) WITH (m={index_param["m"]}, ef_construction={index_param["ef_construction"]}, external={index_param["external"]})')

    def _create_table(self, pg_session, dim):
        try:
            # create table
            index_param = self.case_config.index_param()
            col_type = 'halfvec' if index_param['quant_bits'] == 16 else 'vector'
            pg_session.execute(f'CREATE UNLOGGED TABLE "{self.table_name}" ("{self._primary_field}" INT8, "{self._vector_field}" {col_type}({dim}));')
        except Exception as e:
            log.warning(f"Failed to create pgvector table: {self.table_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> Tuple[int, Exception]:
        try:
            f = StringIO("")
            for i in range(len(metadata)):
                f.write(f"{metadata[i]}\t{embeddings[i]}")
                if i != len(metadata) - 1:
                    f.write('\n')
            f.seek(0)
            pg_session.copy_expert(f'COPY "{self.table_name}" ({self._primary_field}, {self._vector_field}) FROM STDIN', f)
            pg_session.execute(f'VACUUM FULL "{self.table_name}"')
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
        search_param = self.case_config.search_param()
        filter_statement = ''
        vec_id = None
        if filters:
            vec_id = filters.get('id')
            filter_statement = f'WHERE "{self._primary_field}" > {vec_id}'
        
        operator_str = search_param['metric_op']
        statement = f'SELECT "{self._primary_field}" FROM "{self.table_name}" {filter_statement} ORDER BY "{self._vector_field}" {operator_str} \'{query}\' LIMIT {k}'
        self.pg_session.execute(statement)
        res = self.pg_session.fetchall()
        return [row[0] for row in res] 
