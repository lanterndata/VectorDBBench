from abc import ABC, abstractmethod
import pathlib
import time
import logging
from enum import Enum
from typing import Any, Type
from contextlib import contextmanager
from ... import config

from pydantic import BaseModel, validator, SecretStr


log = logging.getLogger(__name__) 

class MetricType(str, Enum):
    L2 = "L2"
    COSINE = "COSINE"
    IP = "IP"

class BoolOpt(str, Enum):
    YES = "YES"
    NO = "NO"

class QuantBitOpt(str, Enum):
    B1 = 1
    I8 = 8
    F16 = 16
    F32 = 32

class IndexType(str, Enum):
    HNSW = "HNSW"
    DISKANN = "DISKANN"
    IVFFlat = "IVF_FLAT"
    GPU_IVFFlat = "GPU_IVF_FLAT"
    IVFSQ8 = "IVF_SQ8"
    IVFSQ8H = "IVF_SQ8H"
    IVFPQ = "IVF_PQ"
    GPU_IVFPQ = "GPU_IVF_PQ"
    Flat = "FLAT"
    AUTOINDEX = "AUTOINDEX"
    ES_HNSW = "hnsw"


class DBConfig(ABC, BaseModel):
    """DBConfig contains the connection info of vector database

    Args:
        db_label(str): label to distinguish different types of DB of the same database.

            MilvusConfig.db_label = 2c8g
            MilvusConfig.db_label = 16c64g
            ZillizCloudConfig.db_label = 1cu-perf
    """

    db_label: str = ""

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError

    @validator("*")
    def not_empty_field(cls, v, field):
        if field.name == "db_label":
            return v
        if isinstance(v, (str, SecretStr)) and len(v) == 0:
            raise ValueError("Empty string!")
        return v


class DBCaseConfig(ABC):
    """Case specific vector database configs, usually uesed for index params like HNSW"""
    @abstractmethod
    def index_param(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def search_param(self) -> dict:
        raise NotImplementedError


class EmptyDBCaseConfig(BaseModel, DBCaseConfig):
    """EmptyDBCaseConfig will be used if the vector database has no case specific configs"""
    null: str | None = None
    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}


class VectorDB(ABC):
    """Each VectorDB will be __init__ once for one case, the object will be copied into multiple processes.

    In each process, the benchmark cases ensure VectorDB.init() calls before any other methods operations

    insert_embeddings, search_embedding, and, optimize will be timed for each call.

    Examples:
        >>> milvus = Milvus()
        >>> with milvus.init():
        >>>     milvus.insert_embeddings()
        >>>     milvus.search_embedding()
    """

    @abstractmethod
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig | None,
        collection_name: str, drop_old: bool = False,
        **kwargs,
    ) -> None:
        """Initialize wrapper around the vector database client.

        Please drop the existing collection if drop_old is True. And create collection
        if collection not in the Vector Database

        Args:
            dim(int): the dimension of the dataset
            db_config(dict): configs to establish connections with the vector database
            db_case_config(DBCaseConfig | None): case specific configs for indexing and searching
            drop_old(bool): whether to drop the existing collection of the dataset.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def config_cls(self) -> Type[DBConfig]:
        raise NotImplementedError


    @classmethod
    @abstractmethod
    def case_config_cls(self, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        raise NotImplementedError


    @abstractmethod
    @contextmanager
    def init(self) -> None:
        """ create and destory connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        raise NotImplementedError

    @abstractmethod
    def load_parquets(
        self, parquet_files: list[str]
        ) -> int:
        """ optional API to directly load parquet files into the data store
        this is called first, before falling back on VectorDB.insert_embeddings

        """
        raise NotImplementedError

    @abstractmethod
    def insert_embeddings(
        self,
        # todo:: this has to take np.array for performance
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        """Insert the embeddings to the vector database. The default number of embeddings for
        each insert_embeddings is 5000.

        Args:
            embeddings(list[list[float]]): list of embedding to add to the vector database.
            metadatas(list[int]): metadata associated with the embeddings, for filtering.
            **kwargs(Any): vector database specific parameters.

        Returns:
            int: inserted data count
        """
        raise NotImplementedError

    @abstractmethod
    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        """Get k most similar embeddings to query vector.

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.
            filters(dict, optional): filtering expression to filter the data while searching.

        Returns:
            list[int]: list of k most similar embeddings IDs to the query embedding.
        """
        raise NotImplementedError

    # TODO: remove
    @abstractmethod
    def optimize(self):
        """optimize will be called between insertion and search in performance cases.

        Should be blocked until the vectorDB is ready to be tested on
        heavy performance cases.

        Time(insert the dataset) + Time(optimize) will be recorded as "load_duration" metric
        Optimize's execution time is limited, the limited time is based on cases.
        """
        raise NotImplementedError

    # TODO: remove
    @abstractmethod
    def ready_to_load(self):
        """ready_to_load will be called before load in load cases.

        should be blocked until the vectordb is ready to be tested on
        heavy load cases.
        """
        raise NotImplementedError

class PgDB(VectorDB):
    def load_parquets(self, parquet_files: list[str]) -> int:
        import pyarrow.dataset as ds
        from pgpq import ArrowToPostgresBinaryEncoder

        total_start = time.time()
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
        self.pg_session.execute(f'DROP TABLE IF EXISTS "{tmp_table_name}"')
        ddl = f"CREATE TEMPORARY TABLE {tmp_table_name} ({typed_cols_joined})"
        self.pg_session.execute(ddl)

        log.debug(f"pg schema {pg_schema}")
        log.debug(f"Assuming underlying postgres table was created with columns: {typed_cols}")

        count = 0
        i = 0
        for batch in dataset.to_batches():
            encoder = ArrowToPostgresBinaryEncoder(dataset.schema)
            log.info(f"batch: {i} batch len: {len(batch)}")
            # Write to buffer
            write_start = time.time()
            self.binary_f.truncate(0)
            self.binary_f.seek(0)
            self.binary_f.write(encoder.write_header())
            self.binary_f.write(encoder.write_batch(batch))
            self.binary_f.write(encoder.finish())
            log.debug(f"Writing batch to buffer took {int(time.time() - write_start)}s")

            # Copy to tmp table
            copy_start = time.time()
            self.binary_f.seek(0)
            self.pg_session.copy_expert(f'COPY "{tmp_table_name}" ({cols_joined}) FROM STDIN WITH (FORMAT BINARY)', self.binary_f)
            log.debug(f"Writing batch to postgres took {int(time.time() - copy_start)}s, total batch processing took: {int(time.time() - write_start)}s")
            count += len(batch)
            i += 1

        # clean buffer
        self.binary_f.seek(0)
        self.binary_f.truncate(0)

        # Copy to actual table
        copy_start = time.time()
        self.pg_session.execute(f'INSERT INTO "{self.table_name}" SELECT * FROM "{tmp_table_name}"')
        self.pg_session.execute(f'ALTER TABLE "{self.table_name}" ADD CONSTRAINT lantern_pk PRIMARY KEY ("{self._primary_field}");')
        self.pg_session.execute(f'ALTER TABLE "{self.table_name}" SET LOGGED')
        self.pg_session.execute(f'DROP TABLE "{tmp_table_name}"')
        log.debug(f"Copy data to main table took {int(time.time() - copy_start)}s, total insert took {int(time.time() - total_start)}s")
        vacuum_start = time.time()
        self.pg_session.execute(f'VACUUM FULL "{self.table_name}"')
        log.debug(f"Table vacuum took {int(time.time() - vacuum_start)}s")

        return count
