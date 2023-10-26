from enum import Enum
from typing import Type

from .api import (
    VectorDB,
    DBConfig,
    DBCaseConfig,
    EmptyDBCaseConfig,
    IndexType,
    MetricType,
)

from .milvus.milvus import Milvus
from .elastic_cloud.elastic_cloud import ElasticCloud
from .pinecone.pinecone import Pinecone
from .weaviate_cloud.weaviate_cloud import WeaviateCloud
from .qdrant_cloud.qdrant_cloud import QdrantCloud
from .zilliz_cloud.zilliz_cloud import ZillizCloud
from .pgvector.pgvector import PgVector
from .lantern.lantern import Lantern
from .redis.redis import Redis
from .chroma.chroma import ChromaClient
from .timescale.timescale import Timescale

class DB(Enum):
    """Database types

    Examples:
        >>> DB.Milvus
        <DB.Milvus: 'Milvus'>
        >>> DB.Milvus.value
        "Milvus"
        >>> DB.Milvus.name
        "Milvus"
    """

    Milvus = "Milvus"
    ZillizCloud = "ZillizCloud"
    Pinecone = "Pinecone"
    ElasticCloud = "ElasticCloud"
    QdrantCloud = "QdrantCloud"
    WeaviateCloud = "WeaviateCloud"
    PgVector = "PgVector"
    Lantern = "Lantern"
    Redis = "Redis"
    Chroma = "Chroma"
    Timescale = "Timescale"


    @property
    def init_cls(self) -> Type[VectorDB]:
        return db2client.get(self)


db2client = {
    DB.Milvus: Milvus,
    DB.ZillizCloud: ZillizCloud,
    DB.WeaviateCloud: WeaviateCloud,
    DB.ElasticCloud: ElasticCloud,
    DB.QdrantCloud: QdrantCloud,
    DB.Pinecone: Pinecone,
    DB.PgVector: PgVector,
    DB.Lantern: Lantern,
    DB.Redis: Redis,
    DB.Chroma: ChromaClient,
    DB.Timescale: Timescale
}

for db in DB:
    assert issubclass(db.init_cls, VectorDB)


__all__ = [
    "DB", "VectorDB", "DBConfig", "DBCaseConfig", "IndexType", "MetricType", "EmptyDBCaseConfig",
]
