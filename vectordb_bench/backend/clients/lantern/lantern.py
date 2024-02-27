"""Wrapper around the Lantern vector database over VectorDB"""

from io import StringIO, BytesIO
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
        if index_param['external']:
            start = time.perf_counter()
            log.info("Start creating external index on table")
            self.create_external_index()
            log.info(f"Finish importing external index into VectorDB, dur={time.perf_counter()-start}")
        else:
            # create vec index
            self._create_index(self.pg_session)

    def ready_to_search(self):
        pass

    def _create_index(self, pg_session):
        index_param = self.case_config.index_param()
        if index_param['external']:
            return

        pg_session.execute(f'CREATE INDEX "{self._index_name}" ON "{self.table_name}" USING lantern_hnsw("{self._vector_field}" {index_param["ops"]}) WITH (m={index_param["m"]}, ef_construction={index_param["ef_construction"]}, ef={index_param["ef"]}, dim={self.dim})')

    def _create_table(self, pg_session, dim):
        try:
            # create table
            pg_session.execute(f'CREATE TABLE "{self.table_name}" ("{self._primary_field}" INT8 PRIMARY KEY, "{self._vector_field}" REAL[{dim}]);')
        except Exception as e:
            log.warning(f"Failed to create lantern table: {self.table_name} error: {e}")
            raise e from None

    def load_parquets(self, parquet_files: list[str]) -> int:
        import pyarrow.dataset as ds
        from pgpq import ArrowToPostgresBinaryEncoder

        # load an arrow dataset
        # arrow can load datasets from partitioned parquet files locally or in S3/GCS
        # it handles buffering, matching globs, etc.
        log.info(f"loading {len(parquet_files)} files")
        log.info(f"loading files: {parquet_files}")
        dataset = ds.dataset(parquet_files)

        # create an encoder object which will do the encoding
        # and give us the expected Postgres table schema
        encoder = ArrowToPostgresBinaryEncoder(dataset.schema)
        # get the expected Postgres destination schema
        # note that this is _not_ the same as the incoming arrow schema
        # and not necessarily the schema of your permanent table
        # instead it's the schema of the data that will be sent over the wire
        # which for example does not have timezones on any timestamps

        # here, since embedding tables just have id and float4[] columns, 
        # it is fine to actualy use this table as the final table
        pg_schema = encoder.schema()
        # assemble ddl for a temporary table
        # it's often a good idea to bulk load into a temp table to:
        # (1) Avoid indexes
        # (2) Stay in-memory as long as possible
        # (3) Be more flexible with types
        #     (you can't load a SMALLINT into a BIGINT column without casting)

        tmp_table_name = "_tmp_parquet_data"
        typed_cols = [f'"{col_name}" {col.data_type.ddl()}' for col_name, col in pg_schema.columns]
        cols = [col_name for col_name, _ in pg_schema.columns]
        cols_joined = ','.join(cols)
        typed_cols_joined = ','.join(typed_cols)

        ddl = f"CREATE UNLOGGED TABLE {tmp_table_name} ({typed_cols_joined})"

        self.pg_session.execute(f"DROP TABLE IF EXISTS {tmp_table_name}")
        self.pg_session.execute(ddl)
        log.debug(f"pg schema {pg_schema}")
        log.debug(f"Assuming underlying postgres table was created with columns: {typed_cols} via a statement equivalent (or columnwise-type-castable) to'{ddl}'")

        self.binary_f.truncate(0)
        self.binary_f.seek(0)
        copy = self.binary_f
        copy.write(encoder.write_header())
        batches = dataset.to_batches()
        count = 0
        for i, batch in enumerate(batches):
            log.info(f"batch: {i} batch len: {len(batch)}")
            b = encoder.write_batch(batch)
            copy.write(b)
            count += len(batch)

        copy.write(encoder.finish())
        self.binary_f.seek(0)
        # todo:: the below can actually run in a separate thread/process, parallel to the above
        log.info(f"Copying dataset into postgres...")
        self.pg_session.copy_expert(f'COPY "{tmp_table_name}" ({cols_joined}) FROM STDIN WITH (FORMAT BINARY)', self.binary_f)
        self.pg_session.execute(f'INSERT INTO "{self.table_name}" SELECT * FROM "{tmp_table_name}"')
        self.pg_session.execute(f'DROP TABLE "{tmp_table_name}"')
        self.pg_session.execute(f'VACUUM FULL "{self.table_name}"')
        return count

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
            'lantern-cli create-index',
            f"-u '{database_url}'",
            f'-t "{self.table_name}"',
            f'-c "{self._vector_field}"',
            f"-m {index_param['m']}",
            f"--ef {index_param['ef']}",
            f"--efc {index_param['ef_construction']}",
            f"-d {self.dim}",
            f"--metric-kind {metric_kind}",
            f"--out {index_file_path}",
            f"--import",
        ])
        log.debug(f"Running external index generation with command {command}")
        _, err = self._run_os_command(command)
        if err:
            raise Exception(err)


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
        
        self.pg_session.execute(f'SET hnsw.init_k={k}')
        self.pg_session.execute(f'SELECT "{self._primary_field}" FROM "{self.table_name}" {filter_statement} ORDER BY "{self._vector_field}" {operator} array{query} LIMIT {k}')
        res = self.pg_session.fetchall()
        return [row[0] for row in res]
        
