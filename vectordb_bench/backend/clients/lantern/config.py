from pydantic import BaseModel, SecretStr
from ..api import BoolOpt, DBConfig, DBCaseConfig, MetricType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"

class LanternConfig(DBConfig):
    user_name: SecretStr = "postgres"
    password: SecretStr = "postgres"
    url: SecretStr = "localhost:5432"
    db_name: str = "lantern"

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        url_str = self.url.get_secret_value()
        return {
            "url" : POSTGRE_URL_PLACEHOLDER%(user_str, pwd_str, url_str, self.db_name)
        }

class LanternIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    M: int | None = 32
    efConstruction: int | None = 128
    ef: int | None = 128
    external_index: BoolOpt | None = BoolOpt.YES
    use_csv: BoolOpt | None = BoolOpt.YES

    def parse_metric(self) -> str: 
        if self.metric_type == MetricType.L2:
            return "dist_l2sq_ops"
        elif self.metric_type == MetricType.COSINE:
            return "dist_cos_ops"
        return "dist_cos_ops"
    
    def parse_metric_fun_str(self) -> str: 
        if self.metric_type == MetricType.L2:
            return "l2sq_dist"
        elif self.metric_type == MetricType.COSINE:
            return "cos_dist"
        return "cos_dist"

    def index_param(self) -> dict:
        return {
            "m": self.M,
            "ef_construction": self.efConstruction,
            "ef": self.ef,
            "external": self.external_index == BoolOpt.YES.value,
            "use_csv": self.use_csv == BoolOpt.YES.value,
            "metric" : self.metric_type,
            "ops" : self.parse_metric()
        }
    
    def search_param(self) -> dict:
        return {
            "metric_fun" : self.parse_metric_fun_str(),
        }
