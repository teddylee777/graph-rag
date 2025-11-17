import os
from collections.abc import Iterator

import pytest
from graph_retriever.adapters import Adapter
from graph_retriever.testing.adapter_tests import AdapterComplianceSuite
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_graph_retriever.transformers import ShreddingTransformer


@pytest.fixture(scope="module")
def connection_string(
    request: pytest.FixtureRequest, enabled_stores: set[str], testcontainers: set[str]
) -> Iterator[str | None]:
    if "pgvector" not in enabled_stores:
        pytest.skip("Pass --stores=pgvector to test PGVector")
        return

    if "pgvector" in testcontainers:
        from testcontainers.postgres import PostgresContainer

        # Use PostgreSQL with pgvector extension
        container = PostgresContainer(
            image="pgvector/pgvector:pg16",
            username="langchain",
            password="langchain",
            dbname="langchain",
        )
        container.start()

        request.addfinalizer(lambda: container.stop())
        # Get connection string from container
        connection = container.get_connection_url(driver="psycopg")
    elif "PGVECTOR_CONNECTION_STRING" in os.environ:
        connection = os.environ["PGVECTOR_CONNECTION_STRING"]
    else:
        # Default connection string for local testing
        connection = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"

    yield connection


class TestPGVector(AdapterComplianceSuite):
    def supports_nested_metadata(self) -> bool:
        # PGVector supports JSONB for metadata, so it can handle nested data
        return True

    @pytest.fixture(scope="class")
    def adapter(
        self,
        connection_string: str,
        animal_embeddings: Embeddings,
        animal_docs: list[Document],
    ) -> Iterator[Adapter]:
        from langchain_postgres import PGVector
        from langchain_graph_retriever.adapters.pgvector import (
            PGVectorAdapter,
        )

        shredder = ShreddingTransformer()

        docs = list(shredder.transform_documents(animal_docs))

        store = PGVector(
            embeddings=animal_embeddings,
            collection_name="animals",
            connection=connection_string,
            use_jsonb=True,
        )

        # Add documents to the store
        store.add_documents(docs)

        yield PGVectorAdapter(
            store, shredder, nested_metadata_fields={"keywords", "tags"}
        )

        # Cleanup: drop the collection
        try:
            store.delete_collection()
        except Exception:
            # If delete fails, it's okay for testing
            pass
