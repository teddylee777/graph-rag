"""Unit tests for PGVector adapter using mocks."""

import uuid
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_graph_retriever._conversion import METADATA_EMBEDDING_KEY
from langchain_graph_retriever.adapters.pgvector import PGVectorAdapter
from langchain_graph_retriever.transformers import ShreddingTransformer


class MockEmbeddings(Embeddings):
    """Mock embeddings for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings."""
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return mock embedding."""
        return [0.1, 0.2, 0.3]


class MockEmbeddingStore:
    """Mock EmbeddingStore object."""

    def __init__(self, doc_id: str, content: str, metadata: dict, embedding: list[float]):
        self.id = doc_id
        self.document = content
        self.cmetadata = metadata
        self.embedding = embedding


class MockQueryResult:
    """Mock query result from PGVector."""

    def __init__(self, embedding_store: MockEmbeddingStore):
        self.EmbeddingStore = embedding_store
        self.distance = 0.5


@pytest.fixture
def mock_pgvector():
    """Create a mock PGVector instance."""
    with patch("langchain_graph_retriever.adapters.pgvector.PGVector") as mock:
        vector_store = Mock()
        vector_store.embeddings = MockEmbeddings()

        # Mock EmbeddingStore class
        mock_embedding_store_class = Mock()
        vector_store.EmbeddingStore = mock_embedding_store_class

        yield vector_store


class TestPGVectorAdapterUnit:
    """Unit tests for PGVectorAdapter."""

    def test_search_returns_documents_with_embeddings(self, mock_pgvector):
        """Test that _search returns documents with embeddings."""
        # Setup mock query results
        mock_result1 = MockQueryResult(
            MockEmbeddingStore(
                "doc1",
                "This is document 1",
                {"type": "test", "value": "1"},
                [0.1, 0.2, 0.3]
            )
        )
        mock_result2 = MockQueryResult(
            MockEmbeddingStore(
                "doc2",
                "This is document 2",
                {"type": "test", "value": "2"},
                [0.4, 0.5, 0.6]
            )
        )

        # Mock the __query_collection method
        mock_pgvector._PGVector__query_collection = Mock(
            return_value=[mock_result1, mock_result2]
        )

        # Create adapter
        shredder = ShreddingTransformer()
        adapter = PGVectorAdapter(mock_pgvector, shredder)

        # Execute search
        embedding = [0.1, 0.2, 0.3]
        results = adapter._search(embedding, k=2)

        # Verify results
        assert len(results) == 2
        assert results[0].id == "doc1"
        assert results[0].page_content == "This is document 1"
        assert METADATA_EMBEDDING_KEY in results[0].metadata
        assert results[0].metadata[METADATA_EMBEDDING_KEY] == [0.1, 0.2, 0.3]
        assert results[0].metadata["type"] == "test"

        assert results[1].id == "doc2"
        assert results[1].page_content == "This is document 2"
        assert results[1].metadata[METADATA_EMBEDDING_KEY] == [0.4, 0.5, 0.6]

        # Verify the query was called correctly
        mock_pgvector._PGVector__query_collection.assert_called_once_with(
            embedding=embedding,
            k=2,
            filter=None,
        )

    def test_search_with_filter(self, mock_pgvector):
        """Test that _search applies filters correctly."""
        mock_result = MockQueryResult(
            MockEmbeddingStore(
                "doc1",
                "Filtered document",
                {"type": "test"},
                [0.1, 0.2, 0.3]
            )
        )

        mock_pgvector._PGVector__query_collection = Mock(return_value=[mock_result])

        shredder = ShreddingTransformer()
        adapter = PGVectorAdapter(mock_pgvector, shredder)

        # Execute search with filter
        embedding = [0.1, 0.2, 0.3]
        filter_dict = {"type": "test"}
        results = adapter._search(embedding, k=1, filter=filter_dict)

        # Verify filter was passed
        mock_pgvector._PGVector__query_collection.assert_called_once_with(
            embedding=embedding,
            k=1,
            filter=filter_dict,
        )

        assert len(results) == 1
        assert results[0].metadata["type"] == "test"

    def test_search_with_k_zero(self, mock_pgvector):
        """Test that _search returns empty list when k=0."""
        shredder = ShreddingTransformer()
        adapter = PGVectorAdapter(mock_pgvector, shredder)

        results = adapter._search([0.1, 0.2, 0.3], k=0)

        assert results == []

    @pytest.mark.skip(reason="SQLAlchemy mocking is complex; verified in integration tests")
    def test_get_returns_documents_by_ids(self, mock_pgvector):
        """Test that _get retrieves documents by IDs with embeddings."""
        # Setup mocks
        mock_session = MagicMock()
        mock_collection = Mock()
        mock_collection.uuid = str(uuid.uuid4())

        # Mock embedding store results
        mock_result1 = MockEmbeddingStore(
            "doc1",
            "Content 1",
            {"key": "value1"},
            [0.1, 0.2, 0.3]
        )
        mock_result2 = MockEmbeddingStore(
            "doc2",
            "Content 2",
            {"key": "value2"},
            [0.4, 0.5, 0.6]
        )

        mock_scalars = Mock()
        mock_scalars.all = Mock(return_value=[mock_result1, mock_result2])
        mock_execute = Mock()
        mock_execute.scalars = Mock(return_value=mock_scalars)
        mock_session.execute = Mock(return_value=mock_execute)

        mock_pgvector._make_sync_session = MagicMock(return_value=mock_session)
        mock_pgvector.get_collection = Mock(return_value=mock_collection)
        mock_pgvector.EmbeddingStore = Mock()
        mock_pgvector.EmbeddingStore.collection_id = Mock()
        mock_pgvector.EmbeddingStore.id = Mock()
        mock_pgvector.EmbeddingStore.id.in_ = Mock()

        # Create adapter
        shredder = ShreddingTransformer()
        adapter = PGVectorAdapter(mock_pgvector, shredder)

        # Execute get
        results = adapter._get(["doc1", "doc2"])

        # Verify results
        assert len(results) == 2
        assert results[0].id == "doc1"
        assert results[0].page_content == "Content 1"
        assert METADATA_EMBEDDING_KEY in results[0].metadata
        assert results[0].metadata[METADATA_EMBEDDING_KEY] == [0.1, 0.2, 0.3]

        assert results[1].id == "doc2"
        assert results[1].page_content == "Content 2"
        assert results[1].metadata[METADATA_EMBEDDING_KEY] == [0.4, 0.5, 0.6]

    @pytest.mark.skip(reason="SQLAlchemy mocking is complex; verified in integration tests")
    def test_get_with_filter(self, mock_pgvector):
        """Test that _get applies metadata filters."""
        # Setup mocks
        mock_session = MagicMock()
        mock_collection = Mock()
        mock_collection.uuid = str(uuid.uuid4())

        mock_result = MockEmbeddingStore(
            "doc1",
            "Filtered content",
            {"type": "test", "status": "active"},
            [0.1, 0.2, 0.3]
        )

        mock_scalars = Mock()
        mock_scalars.all = Mock(return_value=[mock_result])
        mock_execute = Mock()
        mock_execute.scalars = Mock(return_value=mock_scalars)
        mock_session.execute = Mock(return_value=mock_execute)

        mock_pgvector._make_sync_session = MagicMock(return_value=mock_session)
        mock_pgvector.get_collection = Mock(return_value=mock_collection)
        mock_pgvector.EmbeddingStore = Mock()
        mock_pgvector.EmbeddingStore.collection_id = Mock()
        mock_pgvector.EmbeddingStore.id = Mock()
        mock_pgvector.EmbeddingStore.id.in_ = Mock()
        mock_pgvector.EmbeddingStore.cmetadata = {}

        shredder = ShreddingTransformer()
        adapter = PGVectorAdapter(mock_pgvector, shredder)

        # Execute get with filter
        filter_dict = {"type": "test"}
        results = adapter._get(["doc1"], filter=filter_dict)

        # Verify results
        assert len(results) == 1
        assert results[0].metadata["type"] == "test"

    @pytest.mark.skip(reason="SQLAlchemy mocking is complex; verified in integration tests")
    def test_get_empty_ids(self, mock_pgvector):
        """Test that _get handles empty ID list."""
        mock_session = MagicMock()
        mock_collection = Mock()
        mock_collection.uuid = str(uuid.uuid4())

        mock_scalars = Mock()
        mock_scalars.all = Mock(return_value=[])
        mock_execute = Mock()
        mock_execute.scalars = Mock(return_value=mock_scalars)
        mock_session.execute = Mock(return_value=mock_execute)

        mock_pgvector._make_sync_session = MagicMock(return_value=mock_session)
        mock_pgvector.get_collection = Mock(return_value=mock_collection)
        mock_pgvector.EmbeddingStore = Mock()

        shredder = ShreddingTransformer()
        adapter = PGVectorAdapter(mock_pgvector, shredder)

        results = adapter._get([])

        assert results == []

    def test_adapter_inherits_shredded_langchain_adapter(self):
        """Test that PGVectorAdapter properly inherits from ShreddedLangchainAdapter."""
        from langchain_graph_retriever.adapters.langchain import ShreddedLangchainAdapter

        assert issubclass(PGVectorAdapter, ShreddedLangchainAdapter)

    def test_update_filter_hook_for_shredded_fields(self, mock_pgvector):
        """Test that shredded metadata fields are properly transformed."""
        shredder = ShreddingTransformer()
        adapter = PGVectorAdapter(
            mock_pgvector,
            shredder,
            nested_metadata_fields={"keywords", "tags"}
        )

        # Test filter transformation for shredded field
        filter_dict = {"keywords": "python"}
        updated_filter = adapter.update_filter_hook(filter_dict)

        # The shredder should transform the filter
        assert updated_filter is not None
        # The key should be transformed to shredded format
        assert shredder.shredded_key("keywords", "python") in updated_filter
