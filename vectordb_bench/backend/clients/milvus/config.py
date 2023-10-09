from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType, IndexType


class MilvusConfig(DBConfig):
    uri: SecretStr = "http://localhost:19530"

    def to_dict(self) -> dict:
        return {"uri": self.uri.get_secret_value()}


class MilvusIndexConfig(BaseModel):
    """Base config for milvus"""

    index: IndexType
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""

        if self.metric_type == MetricType.COSINE:
            return MetricType.L2.value
        return self.metric_type.value


class AutoIndexConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.AUTOINDEX

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
        }

class HNSWConfig(MilvusIndexConfig, DBCaseConfig):
    M: int
    efConstruction: int
    ef: int | None = None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"M": self.M, "efConstruction": self.efConstruction},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"ef": self.ef},
        }


class DISKANNConfig(MilvusIndexConfig, DBCaseConfig):
    search_list: int | None = None
    index: IndexType = IndexType.DISKANN

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"search_list": self.search_list},
        }


class IVFFlatConfig(MilvusIndexConfig, DBCaseConfig):
    nlist: int
    nprobe: int | None = None
    index: IndexType = IndexType.IVFFlat

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"nlist": self.nlist},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"nprobe": self.nprobe},
        }

class GPUIVFFlatConfig(IVFFlatConfig):
    index: IndexType = IndexType.GPU_IVFFlat

class IVFFSQ8Config(IVFFlatConfig):
    index: IndexType = IndexType.IVFSQ8

class IVFFSQ8HConfig(IVFFlatConfig):
    index: IndexType = IndexType.IVFSQ8H
    
class IVFFPQConfig(IVFFlatConfig):
    index: IndexType = IndexType.IVFPQ
    
class GPUIVFFPQConfig(IVFFSQ8Config):
    pq_m: int | None = None
    index: IndexType = IndexType.GPU_IVFPQ
    
    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"nlist": self.nlist, "m": self.pq_m},
        }
    
class FLATConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.Flat

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {},
        }

_milvus_case_config = {
    IndexType.AUTOINDEX: AutoIndexConfig,
    IndexType.HNSW: HNSWConfig,
    IndexType.DISKANN: DISKANNConfig,
    IndexType.IVFFlat: IVFFlatConfig,
    IndexType.GPU_IVFFlat: GPUIVFFlatConfig,
    IndexType.IVFSQ8: IVFFSQ8Config,
    IndexType.IVFSQ8H: IVFFSQ8HConfig,
    IndexType.IVFPQ: IVFFPQConfig,
    IndexType.GPU_IVFPQ: GPUIVFFPQConfig,
    IndexType.Flat: FLATConfig,
}

