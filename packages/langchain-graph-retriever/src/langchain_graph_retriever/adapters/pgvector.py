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

        # Use the internal method to get documents with embeddings
        # PGVector's similarity_search doesn't return embeddings by default
        # We need to query the database directly to get embeddings
        from sqlalchemy import select

        docs: list[Document] = []

        try:
            with self.vector_store._make_sync_session() as session:
                from langchain_postgres.vectorstores import EmbeddingStore

                # Build the similarity search query with embeddings
                collection = self.vector_store.get_collection(session)
                if not collection:
                    return []

                stmt = select(
                    EmbeddingStore,
                ).filter(
                    EmbeddingStore.collection_id == collection.id
                )

                # Apply metadata filter if provided
                if filter:
                    for key, value in filter.items():
                        stmt = stmt.filter(
                            EmbeddingStore.cmetadata[key].astext == str(value)
                        )

                # Order by similarity (distance)
                stmt = stmt.order_by(
                    EmbeddingStore.embedding.cosine_distance(embedding)
                ).limit(k)

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

        except Exception:
            # Fallback to standard similarity search without embeddings
            # then fetch embeddings separately
            search_results = self.vector_store.similarity_search_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                **kwargs,
            )

            for doc in search_results:
                if doc.id:
                    full_docs = self._get([doc.id], filter=None)
                    if full_docs:
                        docs.append(full_docs[0])
                else:
                    docs.append(doc)

        return docs

    @override
    def _get(
        self, ids: Sequence[str], filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Document]:
        # PGVector stores documents in a table, we need to query by ID
        from sqlalchemy import select

        docs: list[Document] = []

        # Access the underlying collection/table
        store = self.vector_store

        # Build query to get documents by IDs with embeddings
        for doc_id in ids:
            try:
                # Query the store's collection for the document
                # PGVector uses sync operations by default
                with store._make_sync_session() as session:
                    from langchain_postgres.vectorstores import EmbeddingStore

                    stmt = (
                        select(EmbeddingStore)
                        .where(EmbeddingStore.id == doc_id)
                    )

                    if filter:
                        # Apply metadata filters if provided
                        for key, value in filter.items():
                            stmt = stmt.where(
                                EmbeddingStore.cmetadata[key].astext == str(value)
                            )

                    result = session.execute(stmt).scalar_one_or_none()

                    if result:
                        # Convert the result to a Document with embedding
                        doc = Document(
                            id=str(result.id),
                            page_content=result.document,
                            metadata={
                                METADATA_EMBEDDING_KEY: result.embedding,
                                **(result.cmetadata or {}),
                            },
                        )
                        docs.append(doc)
            except Exception:
                # If there's an error fetching a document, skip it
                # This matches the behavior of other adapters
                continue

        return docs
