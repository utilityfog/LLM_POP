from math import sqrt
import os
from typing import Any, Dict, List, Optional, Tuple
import warnings

from langchain_community.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain_openai import OpenAIEmbeddings

from langchain_core.documents import Document
from langchain_core.runnables.config import run_in_executor

import sqlalchemy
from sqlalchemy.orm import Session, relationship

from app.config import EMBEDDING_MODEL, PG_COLLECTION_NAME

# Initialize the embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, disallowed_special=())

# Session ID for current session vector store
CURRENT_SESSION_ID = ""

def set_current_session_id(session_id: str):
    global CURRENT_SESSION_ID
    CURRENT_SESSION_ID = session_id
    print(f"Global session ID set to: {CURRENT_SESSION_ID}")

def get_current_session_id() -> str:
    return CURRENT_SESSION_ID

# Current session vector store
def get_session_vector_store(session_id, pre_delete_collection=False):
    print(f"fucking initializing session vector store via file embedder initialization with session id: {session_id}")
    return PGVector(
        collection_name=session_id,
        connection_string=os.getenv("POSTGRES_URL"),
        embedding_function=embeddings,
        pre_delete_collection=pre_delete_collection
    )
    
def reset_session_vector_store(session_id):
    print(f"resetting session vector store with session id: {session_id}")
    return PGVector(
        collection_name=session_id,
        connection_string=os.getenv("POSTGRES_URL"),
        embedding_function=embeddings,
        pre_delete_collection=True
    ).delete_collection()
    
class PGVectorWithEmbeddings(PGVector):
    def __init__(self, pgvector_instance: PGVector):
        # Initialize with the same attributes as the provided PGVector instance
        super().__init__(
            collection_name=pgvector_instance.collection_name,
            connection_string=os.getenv("POSTGRES_URL"),
            embedding_function=pgvector_instance.embedding_function,
            pre_delete_collection=False
        )
        
    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, List[float], float]]:
        results = self._PGVector__query_collection(embedding=embedding, k=k, filter=filter)

        # Return a list of tuples (Document, embedding, score)
        return [
            (
                Document(
                    page_content=result.EmbeddingStore.document,
                    metadata=result.EmbeddingStore.cmetadata,
                ),
                result.EmbeddingStore.embedding,  # Assuming this is the correct attribute for embedding
                result.distance if self.embedding_function is not None else None,
            )
            for result in results
        ]

# Persistent Vector Store
class PGVectorWithSimilarity(PGVector):
    def _similarity_to_distance(self, similarity: float) -> float:
        """
        Convert a similarity score to a distance metric according to the distance strategy.
        Assumes that the similarity score and distance metrics are inversely related.
        
        - COSINE: Uses 1 - similarity assuming similarity is cosine similarity.
        - EUCLIDEAN: Converts assuming similarity from cosine similarity and vectors are normalized.
        - MAX_INNER_PRODUCT: Uses 1 - similarity, assuming direct conversion for simplicity.
        """
        if self._distance_strategy == DistanceStrategy.COSINE:
            return 1 - similarity
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            # Assuming vectors are normalized, this is an approximation:
            return sqrt(2 * (1 - similarity))
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            # Placeholder, the exact conversion would depend on the nature of how similarity is computed
            return 1 - similarity
        else:
            raise ValueError(f"Unsupported distance strategy: {self._distance_strategy}")
        
    def _results_to_docs(self, docs_and_scores: Any) -> List[Document]:
        """Return docs from docs and scores."""
        return [doc for doc, _ in docs_and_scores]
        
    def similarity_search_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        score_threshold = kwargs.pop("score_threshold", None)

        docs_and_similarities = self._similarity_search_with_relevance_scores(
            embedding, score_threshold, k=k, **kwargs
        )
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                "Relevance scores must be between"
                f" 0 and 1, got {docs_and_similarities}"
            )

        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity < score_threshold
            ]
            if len(docs_and_similarities) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )
        # print(f"fetched context from uploaded file with score: {docs_and_similarities}")
        return self._results_to_docs(docs_and_similarities)
    
    def _similarity_search_with_relevance_scores(
        self,
        embedding: List[float],
        max_similarity_threshold: float,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Default similarity search with relevance scores. Modify if necessary
        in subclass.
        Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = self.similarity_search_with_score(embedding, max_similarity_threshold, k, **kwargs)
        return [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]
    
    async def _asimilarity_search_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Default async similarity search with relevance scores. Modify if necessary
        in subclass.
        Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = await self.asimilarity_search_with_score(embedding, k, **kwargs)
        return [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]
    
    def similarity_search_with_score(
        self,
        embedding: List[float],
        max_similarity_threshold: float,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each.
        """
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, max_similarity_threshold=max_similarity_threshold, k=k, filter=filter
        )
        return docs
    
    async def asimilarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance asynchronously."""

        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        return await run_in_executor(
            None, self.similarity_search_with_score, *args, **kwargs
        )
        
    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        max_similarity_threshold: float,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        results = self.__query_collection(embedding=embedding, max_similarity_threshold=max_similarity_threshold, k=k, filter=filter)

        return self._results_to_docs_and_scores(results)
    
    def __query_collection(
        self,
        embedding: List[float],
        max_similarity_threshold: float,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        with Session(self._bind) as session:
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")

            filter_by = self.EmbeddingStore.collection_id == collection.uuid
            min_distance_allowed = self._similarity_to_distance(max_similarity_threshold)

            if filter is not None:
                filter_clauses = []
                for key, value in filter.items():
                    if isinstance(value, dict):
                        filter_by_metadata = self._create_filter_clause(key, value)
                        if filter_by_metadata is not None:
                            filter_clauses.append(filter_by_metadata)
                    else:
                        filter_by_metadata = self.EmbeddingStore.cmetadata[key].astext == str(value)
                        filter_clauses.append(filter_by_metadata)

                filter_by = sqlalchemy.and_(filter_by, *filter_clauses)

            results = (
                session.query(
                    self.EmbeddingStore,
                    self.distance_strategy(embedding).label("distance")
                )
                .filter(
                    filter_by,
                    self.distance_strategy(embedding) >= min_distance_allowed
                )
                .order_by(sqlalchemy.asc("distance"))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
                )
                .limit(k)
                .all()
            )
            return results
    
def get_persistent_vector_store():
    print(f"fucking initializing persistent vector store {PG_COLLECTION_NAME}")
    return PGVectorWithSimilarity(
        collection_name=PG_COLLECTION_NAME,
        connection_string=os.getenv("POSTGRES_URL"),
        embedding_function=embeddings,
        pre_delete_collection=False
    )

# Session EMBEDDINGS_STORAGE
EMBEDDINGS_STORAGE: {str : List[List[float]]} = {}

# Store and get specific embedding given key
def get_embedding_from_manager(unique_id: str) -> List[List[float]]:
    global EMBEDDINGS_STORAGE
    return EMBEDDINGS_STORAGE.get(unique_id)

def store_embedding(unique_id: str, embedding: List[List[float]]):
    global EMBEDDINGS_STORAGE
    EMBEDDINGS_STORAGE[unique_id] = embedding

# Get or reset embeddings storage
def get_embeddings_storage():
    return EMBEDDINGS_STORAGE

def reset_embeddings_storage():
    global EMBEDDINGS_STORAGE
    EMBEDDINGS_STORAGE = {}
    
### Escalator
# Session EMBEDDINGS_STORAGE
FILE_PATH_DICT: {str : str} = {}

def store_pdf_path(unique_id: str, file_path: str):
    global FILE_PATH_DICT
    FILE_PATH_DICT[unique_id] = file_path
    
def get_file_path(unique_id: str) -> str:
    global FILE_PATH_DICT
    return FILE_PATH_DICT.get(unique_id)

# Session chrome driver
CHROME_DRIVER_DICT = {}

def store_driver(unique_id: str, driver: Any):
    global CHROME_DRIVER_DICT
    CHROME_DRIVER_DICT[unique_id] = driver
    
def get_driver(unique_id: str) -> Any:
    global CHROME_DRIVER_DICT
    return CHROME_DRIVER_DICT.get(unique_id)

def reset_driver():
    global CHROME_DRIVER_DICT
    CHROME_DRIVER_DICT = {}

CHROME_DRIVER_WINDOW_HANDLER_DICT = {}

def store_window_handler(unique_id: str, window_handler: Any):
    global CHROME_DRIVER_WINDOW_HANDLER_DICT
    CHROME_DRIVER_WINDOW_HANDLER_DICT[unique_id] = window_handler
    
def get_window_handler(unique_id: str) -> str:
    global CHROME_DRIVER_WINDOW_HANDLER_DICT
    return CHROME_DRIVER_WINDOW_HANDLER_DICT.get(unique_id)

def reset_window_handlers():
    global CHROME_DRIVER_WINDOW_HANDLER_DICT
    CHROME_DRIVER_WINDOW_HANDLER_DICT = {}
###