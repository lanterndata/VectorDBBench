import chromadb
import logging 
import numpy as np
from contextlib import contextmanager
from typing import Any, Type
from ..api import VectorDB, DBConfig, DBCaseConfig, EmptyDBCaseConfig, IndexType
from .config import ChromaConfig
from chromadb.config import Settings

log = logging.getLogger(__name__)
class ChromaClient(VectorDB):
    """Chroma client for VectorDB. 
    To set up Chroma in docker, see https://docs.trychroma.com/usage-guide
    or the instructions in tests/test_chroma.py

    To change to running in process, modify the HttpClient() in __init__() and init().
    """ 

    def __init__(
            self,
            dim: int,
            db_config: dict,
            db_case_config: DBCaseConfig,
            drop_old: bool = False,

            **kwargs
        ):

        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = 'example2'

        client = chromadb.HttpClient(host=self.db_config["host"], 
                                     port=self.db_config["port"])
        assert client.heartbeat() is not None
        if drop_old:
            try:
                client.reset() # Reset the database
            except:
                drop_old = False
                log.info(f"Chroma client drop_old collection: {self.collection_name}")


    def config_cls() -> Type[DBConfig]:
        return ChromaConfig
    
    def case_config_cls(index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return EmptyDBCaseConfig
    
    @contextmanager
    def init(self) -> None:
        """ create and destory connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        #create connection
        self.client = chromadb.HttpClient(host=self.db_config["host"], 
                                          port=self.db_config["port"])
        
        self.collection = self.client.get_or_create_collection('example2')
        yield
        self.client = None
        self.collection = None

    def ready_to_search(self) -> bool:
        pass

    def ready_to_load(self) -> bool:
        pass

    def optimize(self) -> None:
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        """Insert embeddings into the database.

        Args:
            embeddings(list[list[float]]): list of embeddings
            metadata(list[int]): list of metadata
            kwargs: other arguments

        Returns:
            (int, Exception): number of embeddings inserted and exception if any
        """
        ids=[str(i) for i in metadata]
        metadata = [{"id": int(i)} for i in metadata] 
        if len(embeddings) > 0:
            self.collection.add(embeddings=embeddings, ids=ids, metadatas=metadata)
        return len(embeddings), None
    
    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Search embeddings from the database.
        Args:
            embedding(list[float]): embedding to search
            k(int): number of results to return
            kwargs: other arguments

        Returns:
            Dict {ids: list[list[int]], 
                    embedding: list[list[float]] 
                    distance: list[list[float]]}
        """
        if filters:
            # assumes benchmark test filters of format: {'metadata': '>=10000', 'id': 10000}
            metadata_value = filters.get("metadata")
            id_value = filters.get("id")
            if metadata_value and id_value:
                results = self.collection.query(
                    query_embeddings=query, n_results=k, 
                    where={"$and": [{"id": {"$eq": id_value}}, 
                                    {"id": {"$gt": metadata_value}}
                                    ]}
                )
            elif metadata_value:
                results = self.collection.query(query_embeddings=query, n_results=k, 
                                                where={"id": {"$gt": metadata_value}})
            else:
                results = self.collection.query(query_embeddings=query, n_results=k, 
                                                where={"id": {"$eq": id_value}})
            #return list of id's in results
            return [int(i) for i in results.get('ids')[0]]
        results = self.collection.query(query_embeddings=query, n_results=k)
        return ([int(i) for i in results.get('ids')[0]],)
    
