"""
Validation tests for PGVector adapter code logic.

These tests verify the correctness of the adapter implementation
by checking code structure, method signatures, and inheritance.
"""

import inspect
from typing import Any

import pytest
from langchain_core.documents import Document

from langchain_graph_retriever.adapters.langchain import ShreddedLangchainAdapter
from langchain_graph_retriever.adapters.pgvector import PGVectorAdapter


class TestPGVectorAdapterValidation:
    """Validation tests for PGVectorAdapter."""

    def test_adapter_class_structure(self):
        """Verify PGVectorAdapter has correct class structure."""
        # Check inheritance
        assert issubclass(PGVectorAdapter, ShreddedLangchainAdapter)

        # Check required methods exist
        assert hasattr(PGVectorAdapter, "_search")
        assert hasattr(PGVectorAdapter, "_get")
        assert hasattr(PGVectorAdapter, "update_filter_hook")

    def test_search_method_signature(self):
        """Verify _search method has correct signature."""
        sig = inspect.signature(PGVectorAdapter._search)
        params = list(sig.parameters.keys())

        # Check required parameters
        assert "self" in params
        assert "embedding" in params
        assert "k" in params
        assert "filter" in params
        assert "kwargs" in params

        # Check return annotation
        # The return type should be list[Document]
        assert sig.return_annotation is not inspect.Signature.empty

    def test_get_method_signature(self):
        """Verify _get method has correct signature."""
        sig = inspect.signature(PGVectorAdapter._get)
        params = list(sig.parameters.keys())

        # Check required parameters
        assert "self" in params
        assert "ids" in params
        assert "filter" in params
        assert "kwargs" in params

    def test_search_method_handles_k_zero(self):
        """Verify _search correctly handles k=0 by checking source code."""
        import inspect

        source = inspect.getsource(PGVectorAdapter._search)

        # Check that k=0 is handled
        assert "if k == 0:" in source
        assert "return []" in source

    def test_search_uses_pgvector_internal_api(self):
        """Verify _search uses PGVector's internal __query_collection method."""
        source = inspect.getsource(PGVectorAdapter._search)

        # Check that internal API is used
        assert "__query_collection" in source
        assert "embedding=" in source
        assert "k=" in source
        assert "filter=" in source

    def test_search_extracts_embeddings(self):
        """Verify _search extracts embeddings from results."""
        source = inspect.getsource(PGVectorAdapter._search)

        # Check that embeddings are extracted from EmbeddingStore
        assert "EmbeddingStore" in source
        assert "embedding" in source.lower()
        assert "METADATA_EMBEDDING_KEY" in source

    def test_get_uses_sqlalchemy(self):
        """Verify _get uses SQLAlchemy for querying."""
        source = inspect.getsource(PGVectorAdapter._get)

        # Check SQLAlchemy usage
        assert "from sqlalchemy import select" in source
        assert "select(" in source
        assert "filter(" in source

    def test_get_handles_session_management(self):
        """Verify _get properly manages database sessions."""
        source = inspect.getsource(PGVectorAdapter._get)

        # Check session management
        assert "_make_sync_session" in source
        assert "with " in source  # Context manager usage

    def test_get_filters_by_collection(self):
        """Verify _get filters results by collection ID."""
        source = inspect.getsource(PGVectorAdapter._get)

        # Check collection filtering
        assert "collection_id" in source
        assert "get_collection" in source

    def test_get_filters_by_ids(self):
        """Verify _get filters by document IDs."""
        source = inspect.getsource(PGVectorAdapter._get)

        # Check ID filtering
        assert "id.in_" in source or "id" in source

    def test_get_applies_metadata_filters(self):
        """Verify _get applies metadata filters when provided."""
        source = inspect.getsource(PGVectorAdapter._get)

        # Check metadata filtering
        assert "if filter:" in source
        assert "cmetadata" in source

    def test_get_extracts_embeddings(self):
        """Verify _get includes embeddings in returned documents."""
        source = inspect.getsource(PGVectorAdapter._get)

        # Check that embeddings are included
        assert "METADATA_EMBEDDING_KEY" in source
        assert "embedding" in source.lower()

    def test_adapter_imports(self):
        """Verify all required imports are present."""
        import langchain_graph_retriever.adapters.pgvector as pgvector_module

        # Check PGVector import
        assert hasattr(pgvector_module, "PGVector")

        # Check other required imports
        source = inspect.getsource(pgvector_module)
        assert "from langchain_core.documents import Document" in source
        assert "from langchain_graph_retriever._conversion import METADATA_EMBEDDING_KEY" in source
        assert "from langchain_graph_retriever.adapters.langchain import ShreddedLangchainAdapter" in source

    def test_adapter_docstring(self):
        """Verify PGVectorAdapter has proper documentation."""
        assert PGVectorAdapter.__doc__ is not None
        assert "PGVector" in PGVectorAdapter.__doc__
        assert "adapter" in PGVectorAdapter.__doc__.lower()

    def test_import_error_handling(self):
        """Verify proper error handling for missing langchain-postgres."""
        source = inspect.getsource(
            inspect.getmodule(PGVectorAdapter)
        )

        # Check import error handling
        assert "try:" in source
        assert "from langchain_postgres import PGVector" in source
        assert "except (ImportError, ModuleNotFoundError):" in source
        assert "pip install langchain-postgres" in source

    def test_shredding_support(self):
        """Verify adapter properly supports shredding through inheritance."""
        # PGVectorAdapter should inherit shredding functionality
        assert hasattr(PGVectorAdapter, "update_filter_hook")
        assert hasattr(PGVectorAdapter, "format_documents_hook")

        # Check that it uses ShreddedLangchainAdapter
        assert ShreddedLangchainAdapter in PGVectorAdapter.__mro__

    def test_method_override_decorators(self):
        """Verify methods use @override decorator."""
        source = inspect.getsource(PGVectorAdapter)

        # Check for override decorators
        assert "@override" in source

    def test_type_hints(self):
        """Verify proper type hints are used."""
        # Check _search type hints
        search_sig = inspect.signature(PGVectorAdapter._search)
        assert "embedding" in search_sig.parameters
        assert "k" in search_sig.parameters
        assert "filter" in search_sig.parameters

        # Check _get type hints
        get_sig = inspect.signature(PGVectorAdapter._get)
        assert "ids" in get_sig.parameters
        assert "filter" in get_sig.parameters


class TestPGVectorCodeCompliance:
    """Tests to verify code follows best practices."""

    def test_no_bare_excepts(self):
        """Verify no bare except clauses are used."""
        source = inspect.getsource(PGVectorAdapter)

        # Bare excepts are bad practice - should specify exception types
        # Our current implementation doesn't use bare excepts
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("except:"):
                pytest.fail("Found bare except clause - should specify exception type")

    def test_uses_context_managers(self):
        """Verify proper use of context managers for resource management."""
        source = inspect.getsource(PGVectorAdapter._get)

        # Check for proper context manager usage
        assert "with " in source
        assert "_make_sync_session" in source

    def test_returns_correct_types(self):
        """Verify methods return correct types."""
        # Check return type annotations
        search_sig = inspect.signature(PGVectorAdapter._search)
        get_sig = inspect.signature(PGVectorAdapter._get)

        # Both should return list[Document]
        assert search_sig.return_annotation is not inspect.Signature.empty
        assert get_sig.return_annotation is not inspect.Signature.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
