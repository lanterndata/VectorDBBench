"""Wrapper around the Lantern vector database over VectorDB"""

from io import StringIO, BytesIO
import logging
import time
import subprocess
from contextlib import contextmanager
from typing import Any, Tuple, Type
import psycopg2

from ..api import MetricType, PgDB, DBConfig, DBCaseConfig, IndexType
from .config import LanternConfig, LanternIndexConfig

log = logging.getLogger(__name__) 

class Lantern(PgDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "LanternCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim
        self.f = StringIO("")
        self.binary_f = BytesIO(b"")

        self._index_name = "lantern_index"
        self._primary_field = "id"
        self._vector_field = "embedding"
        self._drop_old = drop_old

        # construct basic units
        pg_engine = psycopg2.connect(db_config['url'])
        pg_engine.autocommit = True
        pg_session = pg_engine.cursor()
        
        # create lantern extension
        pg_session.execute('CREATE EXTENSION IF NOT EXISTS lantern')
        if drop_old:
            log.info(f"Lantern client drop table : {self.table_name}")
            pg_session.execute(f'DROP TABLE IF EXISTS "{self.table_name}" CASCADE')
            pg_session.execute(f'DROP TABLE IF EXISTS "_lantern_internal"."pq_{self.table_name}" CASCADE')
            self._create_table(pg_session, dim)

    
    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return LanternConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return LanternIndexConfig

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
        # self.pg_session.execute("SET work_mem='2GB'")
        # self.pg_session.execute("SET maintenance_work_mem='1GB'")
        yield 
        self.pg_session = None
        self.pg_engine = None 
        del (self.pg_session)
        del (self.pg_engine)
    
    def ready_to_load(self):
        pass

    def optimize(self):
        index_param = self.case_config.index_param()
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
        pg_session.execute(f'''
          SET lantern.external_index_host='{index_param["external_index_host"]}';
          SET lantern.external_index_port={index_param["external_index_port"]};
          SET lantern.external_index_secure={index_param["external_index_secure"]};
        ''')
        pg_session.execute(f'CREATE INDEX "{self._index_name}" ON "{self.table_name}" USING lantern_hnsw("{self._vector_field}" {index_param["ops"]}) WITH (m={index_param["m"]}, ef_construction={index_param["ef_construction"]}, ef={index_param["ef"]}, dim={self.dim}, external={index_param["external"]}, quant_bits={index_param["quant_bits"]})')

    def _create_table(self, pg_session, dim):
        try:
            # create table
            pg_session.execute(f'CREATE UNLOGGED TABLE "{self.table_name}" ("{self._primary_field}" INT8, "{self._vector_field}" REAL[{dim}]);')
        except Exception as e:
            log.warning(f"Failed to create lantern table: {self.table_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> Tuple[int, Exception|None]:
        try:
            self.f.truncate(0)
            self.f.seek(0)
            for i in range(len(metadata)):
                self.f.write(f"{metadata[i]}\t{{{str(embeddings[i])[1:-1]}}}")
                if i != len(metadata) - 1:
                    self.f.write('\n')
            self.f.seek(0)
            self.pg_session.copy_expert(f'COPY "{self.table_name}" ({self._primary_field}, {self._vector_field}) FROM STDIN', self.f)
            self.pg_session.execute(f'VACUUM FULL "{self.table_name}"')
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into lantern table ({self.table_name}), error: {e}")   
            return 0, e

    def _run_os_command(self, command):
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        return output.decode(), error.decode()


    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> Tuple[list[int], float]:
        vec_id = None
        filter_statement = ''
        if filters:
            vec_id = filters.get('id')
            filter_statement = f'WHERE "{self._primary_field}" > {vec_id}'
        operator = '<->'

        index_param = self.case_config.index_param()
        if index_param['metric'] == MetricType.COSINE:
            operator = '<=>'
        
        self.pg_session.execute(f'SET lantern_hnsw.init_k={k}')
        self.pg_session.execute(f'SELECT "{self._primary_field}" FROM "{self.table_name}" {filter_statement} ORDER BY "{self._vector_field}" {operator} array{query} LIMIT {k}')
        res = self.pg_session.fetchall()
        return [row[0] for row in res]
        
