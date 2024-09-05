"""Wrapper around the Pgvector vector database over VectorDB"""

from io import StringIO, BytesIO
import psycopg2
import logging
from contextlib import contextmanager
from typing import Any, Tuple, Type

from ..api import VectorDB, DBConfig, DBCaseConfig, IndexType
from .config import PgVectorConfig, _pgvector_case_config

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
        self.f = StringIO("")
        self.binary_f = BytesIO(b"")

        self._index_name = "pqvector_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        # construct basic units
        pg_engine = psycopg2.connect(db_config['url'])
        pg_engine.autocommit = True
        pg_session = pg_engine.cursor()
        
        # create lantern extension
        pg_session.execute('CREATE EXTENSION IF NOT EXISTS lantern')
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
            pg_session.execute(f'CREATE TABLE "{self.table_name}" ("{self._primary_field}" INT PRIMARY KEY, "{self._vector_field}" {col_type}({dim}));')
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
