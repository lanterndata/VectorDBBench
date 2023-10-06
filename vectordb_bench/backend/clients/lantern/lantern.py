"""Wrapper around the Lantern vector database over VectorDB"""

import logging
import time
import subprocess
from contextlib import contextmanager
from typing import Any, Tuple, Type

from ..api import MetricType, VectorDB, DBConfig, DBCaseConfig, IndexType
from .config import LanternConfig, LanternIndexConfig
from sqlalchemy import (
    create_engine,
    insert,
    Index,
    Table,
    text,
    Column,
    Integer,
    ARRAY,
    REAL
)
from sqlalchemy.orm import (
    declarative_base, 
    Session
)

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
        pg_engine = create_engine(**self.db_config)
        
        # create lantern extension
        with pg_engine.connect() as conn: 
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS lantern'))
            if drop_old:
                log.info(f"Lantern client drop table : {self.table_name}")
                conn.execute(text(f'DROP TABLE IF EXISTS "{self.table_name}" CASCADE'))
            conn.commit()
        
        # redefined engine so metadata cache will be cleared after drop
        pg_engine = create_engine(**self.db_config)
        Base = declarative_base()
        pq_metadata = Base.metadata
        pq_metadata.reflect(pg_engine) 

        self.pg_table = self._get_table_schema(pq_metadata)

        if drop_old and self.table_name in pq_metadata.tables:
            self._create_table(dim, pg_engine)

    
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
            Column(self._vector_field, ARRAY(REAL)),
            extend_existing=True
        )
    
    def _create_index(self, pg_engine):
        index_param = self.case_config.index_param()
        if index_param['external']:
            return

        index = Index(self._index_name, self.pg_table.c.embedding,
            postgresql_using='hnsw',
            postgresql_with={'m': index_param["m"], 'ef_construction': index_param['ef_construction'], 'ef': index_param['ef'], 'dim': self.dim},
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
            log.warning(f"Failed to create lantern table: {self.table_name} error: {e}")
            raise e from None

    def copy_embeddings_from_csv(self, csv_path):
        self.pg_session.execute(text(f'COPY "{self.table_name}" ("{self._primary_field}", "{self._vector_field}") FROM \'{csv_path}\' WITH CSV'))
        self.pg_session.commit()
        return None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        try:
            if self._drop_old:
                return len(metadata), None
            items = [dict(id = metadata[i], embedding=embeddings[i]) for i in range(len(metadata))]
            self.pg_session.execute(insert(self.pg_table), items)
            self.pg_session.commit()
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
        if self.pg_table.columns.embedding.index:
            log.info('Index already exists dropping...')
            self.pg_session.execute(text(f'DROP INDEX {self._index_name} ON {self.table_name}'))
            self.pg_session.commit()

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


        self.pg_session.execute(text(f'CREATE INDEX "{self._index_name}" ON "{self.table_name}" USING hnsw("{self._vector_field}") WITH (_experimental_index_path=\'{index_file_path}\')'))
        self.pg_session.commit()


    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> Tuple[list[int], float]:
        assert self.pg_table is not None
        vec_id = None
        filter_statement = ''
        if filters:
            vec_id = filters.get('id')
            filter_statement = f'WHERE "{self._primary_field}" > {vec_id}'
        
        statement = text(f'SELECT "{self._primary_field}" FROM "{self.pg_table}" {filter_statement} ORDER BY "{self._vector_field}" <-> array{query} LIMIT {k}')
        s = time.perf_counter()
        res = self.pg_session.execute(statement)
        return [row[0] for row in res.fetchall()], time.perf_counter() - s
        
