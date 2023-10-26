from pydantic import SecretStr
from ..api import DBConfig

class TimescaleConfig(DBConfig):
    service_url: SecretStr

    def to_dict(self) -> dict:
        return {
            "url" : self.service_url.get_secret_value()
        }
