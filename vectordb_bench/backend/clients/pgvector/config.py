from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType, IndexType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"

class PgVectorConfig(DBConfig):
    user_name: SecretStr = "postgres"
    password: SecretStr
    url: SecretStr
    db_name: str

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        url_str = self.url.get_secret_value()
        return {
            "url" : POSTGRE_URL_PLACEHOLDER%(user_str, pwd_str, url_str, self.db_name)
        }

class PgVectorIndexConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.IVFFlat
    metric_type: MetricType | None = None
    lists: int | None = 1000
    probes: int | None = 10

    def parse_metric(self) -> str: 
        if self.metric_type == MetricType.L2:
            return "vector_l2_ops"
        elif self.metric_type == MetricType.IP:
            return "vector_ip_ops"
        return "vector_cosine_ops"
    
    def parse_metric_fun_str(self) -> str: 
        if self.metric_type == MetricType.L2:
            return "l2_distance"
        elif self.metric_type == MetricType.IP:
            return "max_inner_product"
        return "cosine_distance"
        
    def parse_metric_op_str(self) -> str: 
        if self.metric_type == MetricType.L2:
            return "<->"
        elif self.metric_type == MetricType.COSINE:
            return "<=>"
        return "<=>"



class PgVectorIVFFlatConfig(PgVectorIndexConfig):
    index = IndexType.IVFFlat
    
    def index_param(self) -> dict:
        return {
            "lists" : self.lists,
            "metric" : self.parse_metric()
        }
    
    def search_param(self) -> dict:
        return {
            "probes" : self.probes,
            "metric_fun" : self.parse_metric_fun_str(),
            "metric_op" : self.parse_metric_op_str()
        }
    
class PgVectorHNSWConfig(PgVectorIndexConfig):
    index = IndexType.HNSW
    efConstruction: int | None = 128
    m: int | None = 32
    ef: int | None = 128
    def index_param(self) -> dict:
        return {
            "metric" : self.parse_metric(),
            "m" : self.m,
            "ef_construction" : self.efConstruction
        }
    
    def search_param(self) -> dict:
        return {
            "ef" : self.ef,
            "metric_fun" : self.parse_metric_fun_str(),
            "metric_op" : self.parse_metric_op_str()
        }
    
_pgvector_case_config = {
    IndexType.IVFFlat: PgVectorIVFFlatConfig,
    IndexType.HNSW: PgVectorHNSWConfig,
}

