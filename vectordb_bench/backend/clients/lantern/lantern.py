"""Wrapper around the Lantern vector database over VectorDB"""

import logging
import time
import subprocess
from contextlib import contextmanager
from typing import Any, Tuple, Type
import psycopg2

from ..api import MetricType, VectorDB, DBConfig, DBCaseConfig, IndexType
from .config import LanternConfig, LanternIndexConfig

log = logging.getLogger(__name__) 

class Lantern(VectorDB):
    """ Use SQLAlchemy instructions"""
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
        yield 
        self.pg_session = None
        self.pg_engine = None 
        del (self.pg_session)
        del (self.pg_engine)
    
    def ready_to_load(self):
        pass

    def optimize(self):
        index_param = self.case_config.index_param()
        if index_param['external']:
            start = time.perf_counter()
            log.info("Start creating external index on table")
            self.create_external_index()
            log.info(f"Finish importing external index into VectorDB, dur={time.perf_counter()-start}")

    def ready_to_search(self):
        pass

    def _create_index(self, pg_session):
        index_param = self.case_config.index_param()
        if index_param['external']:
            return

        pg_session.execute(f'DROP INDEX IF EXISTS "{self._index_name}"')
        pg_session.execute(f'CREATE INDEX "{self._index_name}" ON "{self.table_name}" USING hnsw("{self._vector_field}") WITH (m={index_param["m"]}, ef_construction={index_param["ef_construction"]}, ef={index_param["ef"]}, dim={self.dim})')

    def _create_table(self, pg_session, dim):
        try:
            # create table
            pg_session.execute(f'CREATE TABLE "{self.table_name}" ("{self._primary_field}" INT PRIMARY KEY, "{self._vector_field}" REAL[{dim}]);')
            # create vec index
            self._create_index(pg_session)
        except Exception as e:
            log.warning(f"Failed to create lantern table: {self.table_name} error: {e}")
            raise e from None

    def copy_embeddings_from_csv(self, csv_path):
        self.pg_session.execute(f'COPY "{self.table_name}" ("{self._primary_field}", "{self._vector_field}") FROM \'{csv_path}\' WITH CSV')
        return None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> Tuple[int, Exception|None]:
        try:
            if self._drop_old:
                return len(metadata), None
            items = [f"({i}, ARRAY{metadata[i]})" for i in range(len(metadata))]
            self.pg_session.execute(f'INSERT INTO "{self.table_name}" ("{self._primary_field}", "{self._vector_field}") VALUES ({",".join(items)})')
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into lantern table ({self.table_name}), error: {e}")   
            return 0, e

    def _run_os_command(self, command):
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        return output.decode(), error.decode()

    def create_external_index(self):
        self.pg_session.execute(f'DROP INDEX IF EXISTS {self._index_name}')

        if not self.external_index_dir.exists():
            log.info(f"external index file path not exist, creating it: {self.external_index_dir}")
            self.external_index_dir.mkdir(parents=True)

        # Create external index and save to file
        database_url = self.db_config['url']
        index_param = self.case_config.index_param()
        index_file_path = self.external_index_dir.joinpath('index.usearch').absolute()
        metric_kind = 'l2sq'

        if index_param['metric'] == MetricType.COSINE:
            metric_kind = 'cos'

        command = ' '.join([
            'lantern-create-index',
            f'-u "{database_url}"',
            f'-t "{self.table_name}"',
            f'-c "{self._vector_field}"',
            f"-m {index_param['m']}",
            f"--ef {index_param['ef']}",
            f"--efc {index_param['ef_construction']}",
            f"-d {self.dim}",
            f"--metric-kind {metric_kind}",
            f"--out {index_file_path}",
        ])
        _, err = self._run_os_command(command)
        if err:
            raise Exception(err)


        self.pg_session.execute(f'CREATE INDEX "{self._index_name}" ON "{self.table_name}" USING hnsw("{self._vector_field}") WITH (_experimental_index_path=\'{index_file_path}\')')


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
        
        self.pg_session.execute(f'SELECT "{self._primary_field}" FROM "{self.table_name}" {filter_statement} ORDER BY "{self._vector_field}" <-> array{query} LIMIT {k}')
        res = self.pg_session.fetchall()
        return ([row[0] for row in res], 0)
        
