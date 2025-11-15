from collections.abc import Iterator

import pytest
from graph_retriever.adapters import Adapter
from graph_retriever.testing.adapter_tests import AdapterComplianceSuite
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_graph_retriever.transformers import ShreddingTransformer


class TestPGVector(AdapterComplianceSuite):
    def supports_nested_metadata(self) -> bool:
        # PGVector supports JSONB for metadata, so it can handle nested data
        return True

    @pytest.fixture(scope="class")
    def adapter(
        self,
        enabled_stores: set[str],
        animal_embeddings: Embeddings,
        animal_docs: list[Document],
    ) -> Iterator[Adapter]:
        if "pgvector" not in enabled_stores:
            pytest.skip("Pass --stores=pgvector to test PGVector")

        from langchain_postgres import PGVector
        from langchain_graph_retriever.adapters.pgvector import (
            PGVectorAdapter,
        )

        shredder = ShreddingTransformer()

        docs = list(shredder.transform_documents(animal_docs))

        # Connection string for test database
        # This assumes a PostgreSQL instance with pgvector is running
        connection = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"

        store = PGVector(
            embeddings=animal_embeddings,
            collection_name="animals",
            connection=connection,
            use_jsonb=True,
        )

        # Add documents to the store
        store.add_documents(docs)

        yield PGVectorAdapter(
            store, shredder, nested_metadata_fields={"keywords", "tags"}
        )

        # Cleanup: drop the collection
        store.delete_collection()
