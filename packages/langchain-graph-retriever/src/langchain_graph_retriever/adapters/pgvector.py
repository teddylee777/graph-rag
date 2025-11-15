"""Provides an adapter for PGVector vector store integration."""

from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from typing_extensions import override

from langchain_graph_retriever._conversion import METADATA_EMBEDDING_KEY
from langchain_graph_retriever.adapters.langchain import ShreddedLangchainAdapter

try:
    from langchain_postgres import PGVector
except (ImportError, ModuleNotFoundError):
    msg = "please `pip install langchain-postgres`"
    raise ImportError(msg)


class PGVectorAdapter(ShreddedLangchainAdapter[PGVector]):
    """
    Adapter for [PGVector](https://github.com/pgvector/pgvector) vector store.

    This adapter integrates the LangChain PGVector vector store with the
    graph retriever system, allowing for similarity search and document retrieval
    using PostgreSQL with the pgvector extension.

    Parameters
    ----------
    vector_store :
        The PGVector vector store instance.
    shredder: ShreddingTransformer, optional
        An instance of the ShreddingTransformer used for doc insertion.
        If not passed then a default instance of ShreddingTransformer is used.
    """

    @override
    def _search(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        if k == 0:
            return []

        # Use PGVector's internal query method to get results with embeddings
        results = self.vector_store._PGVector__query_collection(
            embedding=embedding,
            k=k,
            filter=filter,
        )

        docs: list[Document] = []
        for result in results:
            # Extract embedding from the result
            # PGVector returns results with EmbeddingStore objects
            doc = Document(
                id=str(result.EmbeddingStore.id),
                page_content=result.EmbeddingStore.document,
                metadata={
                    METADATA_EMBEDDING_KEY: result.EmbeddingStore.embedding,
                    **(result.EmbeddingStore.cmetadata or {}),
                },
            )
            docs.append(doc)

        return docs

    @override
    def _get(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Document]:
        from sqlalchemy import select

        docs: list[Document] = []

        with self.vector_store._make_sync_session() as session:
            collection = self.vector_store.get_collection(session)

            # Build query to get documents by IDs with embeddings
            filter_by = [self.vector_store.EmbeddingStore.collection_id == collection.uuid]

            # Add ID filter
            filter_by.append(self.vector_store.EmbeddingStore.id.in_(ids))

            # Add metadata filters if provided
            if filter:
                for key, value in filter.items():
                    filter_by.append(
                        self.vector_store.EmbeddingStore.cmetadata[key].astext == str(value)
                    )

            stmt = select(self.vector_store.EmbeddingStore).filter(*filter_by)

            results = session.execute(stmt).scalars().all()

            for result in results:
                doc = Document(
                    id=str(result.id),
                    page_content=result.document,
                    metadata={
                        METADATA_EMBEDDING_KEY: result.embedding,
                        **(result.cmetadata or {}),
                    },
                )
                docs.append(doc)

        return docs
