# CLAUDE.md - AI Assistant Guide for Graph RAG

**Last Updated:** 2025-11-15
**Repository:** https://github.com/datastax/graph-rag
**Purpose:** Guide for AI assistants working on the Graph RAG codebase

---

## 1. Project Overview

### What is Graph RAG?

Graph RAG is a Python library that **combines vector search (unstructured similarity) with graph traversal (structured metadata relationships)** for enhanced Retrieval-Augmented Generation (RAG). It allows applications to traverse existing vector stores while maintaining relationships defined in document metadata.

### Key Features
- **Hybrid retrieval**: Vector similarity + graph traversal
- **Vector store agnostic**: Works with existing vector stores (AstraDB, Cassandra, Chroma, OpenSearch, etc.)
- **Multiple traversal strategies**: Eager (breadth-first), MMR (diversity), Scored (custom scoring)
- **LangChain integration**: Compatible with LangChain's retriever framework
- **Document transformers**: NER, keyword extraction, HTML parsing, shredding, etc.

### Architecture Components

1. **`graph-retriever`** (Core Library)
   - Pure Python graph traversal engine
   - Store-agnostic adapter pattern
   - Traversal strategies and edge definitions
   - No LangChain dependencies

2. **`langchain-graph-retriever`** (LangChain Integration)
   - LangChain-compatible `GraphRetriever` class
   - Vector store adapters (AstraDB, Cassandra, Chroma, OpenSearch)
   - Document transformers (Spacy, KeyBERT, GliNER, HTML, etc.)
   - Node ↔ Document conversion utilities

3. **`graph-rag-example-helpers`** (Examples & Datasets)
   - Pre-loaded datasets (animals, astrapy, wikimultihop)
   - Example code generation utilities
   - Environment variable handling

---

## 2. Repository Structure

```
/home/user/graph-rag/
├── packages/                          # Monorepo packages (UV workspace)
│   ├── graph-retriever/              # Core graph traversal engine
│   │   ├── src/graph_retriever/
│   │   │   ├── adapters/            # Base adapter + in-memory implementation
│   │   │   ├── edges/               # Edge definitions and metadata extraction
│   │   │   ├── strategies/          # Traversal strategies (Eager, MMR, Scored)
│   │   │   ├── utils/               # Math, batching, merging utilities
│   │   │   ├── traversal.py         # Main traverse() and atraverse() functions
│   │   │   ├── types.py             # Node dataclass
│   │   │   └── testing/             # Testing infrastructure
│   │   └── tests/                   # Unit tests
│   │
│   ├── langchain-graph-retriever/    # LangChain integration
│   │   ├── src/langchain_graph_retriever/
│   │   │   ├── adapters/            # Vector store adapters
│   │   │   ├── transformers/        # Document transformers
│   │   │   ├── graph_retriever.py   # Main GraphRetriever class
│   │   │   ├── document_graph.py    # Graph utilities (networkx)
│   │   │   └── _conversion.py       # Node ↔ Document conversion
│   │   └── tests/                   # Integration tests
│   │
│   └── graph-rag-example-helpers/    # Examples & datasets
│       └── src/graph_rag_example_helpers/
│
├── docs/                             # MkDocs documentation
│   ├── guide/                        # User guides
│   ├── examples/                     # Jupyter notebooks
│   ├── reference/                    # Auto-generated API docs
│   └── blog/                         # Blog posts
│
├── data/                             # Test datasets (JSONL format)
├── .github/                          # CI/CD workflows
├── scripts/                          # Utility scripts
├── pyproject.toml                    # Root workspace config
├── uv.lock                           # Dependency lock file
└── mkdocs.yml                        # Documentation config
```

### Key Files by Purpose

| Purpose | File Path | Description |
|---------|-----------|-------------|
| **Core Traversal** | `packages/graph-retriever/src/graph_retriever/traversal.py:14-62` | Main `traverse()` function (426 lines) |
| **LangChain Integration** | `packages/langchain-graph-retriever/src/langchain_graph_retriever/graph_retriever.py:24-201` | `GraphRetriever` class |
| **Node Definition** | `packages/graph-retriever/src/graph_retriever/types.py` | `Node` dataclass with depth, embedding, edges |
| **Edge Extraction** | `packages/graph-retriever/src/graph_retriever/edges/metadata.py` | `MetadataEdgeFunction` for extracting edges |
| **Strategy Base** | `packages/graph-retriever/src/graph_retriever/strategies/base.py` | Abstract `Strategy` class |
| **Adapter Base** | `packages/graph-retriever/src/graph_retriever/adapters/base.py` | Abstract `Adapter` class |
| **Workspace Config** | `pyproject.toml` | UV workspace, pytest, ruff, coverage settings |

---

## 3. Development Environment Setup

### Prerequisites

- **Python**: 3.10+ (project uses 3.12.8)
- **UV**: >= 0.5.0 (package manager)
- **Git**: For version control

### Initial Setup

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/datastax/graph-rag.git
cd graph-rag

# Install the project Python version
uv python install

# Create virtual environment and install dev dependencies
uv sync

# Install all dependencies (including docs and all extras)
uv run poe sync
```

### Environment Variables

Create a `.env` file based on `.env.template`:

```bash
# Required for AstraDB testing
ASTRA_DB_APPLICATION_TOKEN=<your-token>
ASTRA_DB_API_ENDPOINT=<your-endpoint>

# Required for documentation notebooks
OPENAI_API_KEY=<your-openai-key>
```

### Recommended Shell Alias

Add to your `.bashrc` or `.zshrc`:

```bash
urp() {
    uv run poe "$@"
}
```

Then use `urp lint` instead of `uv run poe lint`.

---

## 4. Code Organization and Key Modules

### 4.1 Core Library (`graph-retriever`)

#### Traversal Flow

```python
# Entry point: packages/graph-retriever/src/graph_retriever/traversal.py:14
def traverse(query, edges, strategy, store, metadata_filter=None, initial_root_ids=(), store_kwargs={})
```

**Traversal Process:**
1. **Initial Search**: Vector similarity search using `store.search(query)`
2. **Edge Extraction**: Extract edges from metadata using `EdgeFunction`
3. **Graph Traversal**: Iteratively traverse graph using `Strategy`
4. **Node Selection**: Apply strategy to select nodes (Eager/MMR/Scored)
5. **Return Results**: Top-k nodes with depth and similarity scores

#### Key Classes

**Node** (`types.py`):
```python
@dataclass
class Node:
    id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any]
    depth: int
    edges: dict[str, set[str]]  # Edge type -> target IDs
    score: float | None = None
```

**Strategy** (`strategies/base.py`):
- Abstract base class defining traversal behavior
- `iteration(tracker, depth)`: Select nodes at each depth
- Implementations: `Eager`, `MMR`, `Scored`

**Adapter** (`adapters/base.py`):
- Abstract interface for vector stores
- `search(query, k, filter)`: Similarity search
- `get_by_ids(ids, filter)`: Retrieve by IDs
- `asearch()`, `aget_by_ids()`: Async versions

**EdgeFunction** (`edges/_base.py`):
- Extract edges from node metadata
- `MetadataEdgeFunction`: Extract from field patterns
- Edge specs: `("source_field", "target_field")`

### 4.2 LangChain Integration (`langchain-graph-retriever`)

#### GraphRetriever

```python
# packages/langchain-graph-retriever/src/langchain_graph_retriever/graph_retriever.py:24
class GraphRetriever(BaseRetriever):
    store: Adapter | VectorStore
    edges: list[EdgeSpec] | EdgeFunction = []
    strategy: Strategy = Eager()
```

- **Pydantic-based**: Uses Pydantic for validation
- **Auto-adapter inference**: Detects vector store type and selects adapter
- **LangChain compatible**: Implements `BaseRetriever` interface
- **Flexible configuration**: Supports extra kwargs passed to strategy

#### Vector Store Adapters

| Adapter | File | Notes |
|---------|------|-------|
| AstraDB | `adapters/astra.py` | Batching optimizations, advanced filtering |
| Cassandra | `adapters/cassandra.py` | CassIO integration |
| Chroma | `adapters/chroma.py` | Standard Chroma support |
| OpenSearch | `adapters/open_search.py` | OpenSearch integration |
| In-Memory | `adapters/in_memory.py` | Testing/development |

**Adapter Inference** (`adapters/inference.py:93`):
- Automatically selects adapter based on store class hierarchy
- Example: `AstraDBVectorStore` → `AstraDBAdapter`

#### Document Transformers

| Transformer | File | Purpose |
|-------------|------|---------|
| SpacyNER | `transformers/spacy.py` | Named Entity Recognition |
| KeyBERT | `transformers/keybert.py` | Keyword extraction |
| GliNER | `transformers/gliner.py` | Zero-shot entity extraction |
| HTML | `transformers/html.py` | HTML parsing (BeautifulSoup) |
| Shredding | `transformers/shredding.py` | Split large documents |
| Parent | `transformers/parent.py` | Parent-child relationships |

### 4.3 Example Helpers (`graph-rag-example-helpers`)

**Datasets** (`datasets/`):
```python
from graph_rag_example_helpers.datasets import animals, astrapy, wikimultihop

# Load pre-configured datasets
animal_docs = animals.fetch()
astrapy_docs = astrapy.fetch()
```

**Environment** (`env.py`):
- Handles `.env` loading
- Validates required environment variables

---

## 5. Development Workflow

### 5.1 Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/AmazingFeature
   ```

2. **Make your changes** in the appropriate package:
   - Core logic → `packages/graph-retriever/`
   - LangChain integration → `packages/langchain-graph-retriever/`
   - Examples → `packages/graph-rag-example-helpers/`

3. **Run quality checks**:
   ```bash
   uv run poe lint-fix      # Auto-fix formatting and lints
   uv run poe type-check    # Run mypy type checking
   ```

4. **Run tests**:
   ```bash
   uv run poe test          # Unit tests (in-memory stores)
   uv run poe test-all      # All tests (all stores, requires credentials)
   ```

5. **Update documentation** (if needed):
   ```bash
   uv run poe docs-serve    # Live preview
   ```

### 5.2 Common Development Tasks

#### Code Quality

```bash
# Format code
uv run poe fmt-fix

# Lint and fix
uv run poe lint-fix

# Type check (mypy)
uv run poe type-check

# Check dependencies
uv run poe dep-check

# Run all checks
uv run poe lint
```

#### Testing

```bash
# Test core library only
uv run poe test-gr

# Test langchain integration only
uv run poe test-lgr

# Test both (in-memory stores)
uv run poe test

# Test all stores (requires env vars)
uv run poe test-all

# Test notebooks
uv run poe test-nb

# Generate coverage report
uv run poe coverage
```

#### Documentation

```bash
# Serve docs locally (with live reload)
uv run poe docs-serve

# Build docs (strict mode)
uv run poe docs-build
```

#### Dependency Management

```bash
# Update lock file
uv run poe lock-fix

# Check lock file consistency
uv run poe lock-check

# Sync all dependencies
uv run poe sync
```

#### Notebooks

```bash
# Strip output from notebooks (auto-format)
uv run poe nbstripout

# Check notebooks are stripped
uv run poe nbstripout-check
```

---

## 6. Testing Guidelines

### 6.1 Test Organization

**Core Library** (`packages/graph-retriever/tests/`):
- Unit tests for adapters, strategies, edges, utils
- In-memory adapter used for testing
- No external dependencies

**LangChain Integration** (`packages/langchain-graph-retriever/tests/`):
- Integration tests with multiple vector stores
- Adapter compliance tests
- Transformer tests (marked with `@pytest.mark.extra`)
- Uses testcontainers for external stores

### 6.2 Test Fixtures

**Animal Dataset** (`conftest.py`):
```python
@pytest.fixture
def animal_docs():
    # 26 animal documents with keywords, habitat, diet metadata
    # Used for testing graph traversal
    pass
```

**Store Selection**:
```python
# Test with in-memory store only (default)
pytest

# Test with all stores
pytest --stores=all

# Test with specific stores
pytest --stores=chroma --stores=opensearch
```

**Extra Dependencies**:
```python
# Skip tests requiring optional dependencies
pytest

# Run all tests including extras
pytest --runextras
```

### 6.3 Test Patterns

**Sync/Async Dual Testing**:
```python
@pytest.mark.parametrize("sync_or_async", ["sync", "async"])
async def test_traversal(sync_or_async):
    if sync_or_async == "sync":
        result = traverse(query, ...)
    else:
        result = await atraverse(query, ...)
```

**Adapter Compliance Suite**:
```python
# All adapters must pass the same compliance tests
class TestAstraDBAdapter(AdapterComplianceSuite):
    # Inherit standard adapter tests
    pass
```

**Testcontainers**:
- Cassandra 5.0.1
- Chroma 0.5.23
- OpenSearch 2.18.0
- Automatically started/stopped for tests

### 6.4 Writing Tests

**DO:**
- ✅ Test both sync and async code paths
- ✅ Use the `animal_docs` fixture for graph traversal tests
- ✅ Mark tests requiring optional dependencies with `@pytest.mark.extra`
- ✅ Test edge cases (empty results, invalid IDs, etc.)
- ✅ Use descriptive test names: `test_<functionality>_<scenario>`

**DON'T:**
- ❌ Don't commit notebooks with output (run `uv run poe nbstripout`)
- ❌ Don't hardcode credentials (use environment variables)
- ❌ Don't skip cleanup in tests (use fixtures with yield)
- ❌ Don't test implementation details (test behavior)

---

## 7. Code Style and Conventions

### 7.1 Formatting and Linting

**Tool: Ruff** (configured in `pyproject.toml`)

- **Format**: `ruff format` (Black-compatible)
- **Lint**: `ruff check` (Flake8 + pylint + isort)
- **Line length**: 88 characters (Black default)
- **Import sorting**: Automatic (isort rules)

**Enabled Rules**:
- `E`: pycodestyle errors
- `F`: pyflakes
- `I`: isort (import sorting)
- `T201`: print statements (avoid in production code)
- `D`: pydocstyle (NumPy convention)
- `W`: pycodestyle warnings
- `UP`: pyupgrade (modern Python syntax)
- `DOC`: docstring linting

**Ignored Rules**:
- `D100`: Missing module docstring
- `D104`: Missing package docstring
- `D107`: Missing `__init__` docstring

### 7.2 Type Hints

**Tool: Mypy** (strict mode)

```python
# Use type hints for all functions
def traverse(
    query: str,
    *,
    edges: list[EdgeSpec] | EdgeFunction,
    strategy: Strategy,
    store: Adapter,
    metadata_filter: dict[str, Any] | None = None,
) -> list[Node]:
    ...

# Use from __future__ import annotations for deferred evaluation
from __future__ import annotations

# Use typing-extensions for compatibility
from typing_extensions import Self
```

### 7.3 Docstrings

**Style: NumPy** (configured in `pyproject.toml`)

```python
def traverse(query: str, ...) -> list[Node]:
    """
    Perform a graph traversal to retrieve nodes for a specific query.

    Parameters
    ----------
    query :
        The query string for the traversal.
    edges :
        A list of EdgeSpec for use in creating a MetadataEdgeFunction,
        or an EdgeFunction.
    strategy :
        The traversal strategy that defines how nodes are discovered.

    Returns
    -------
    :
        Nodes returned by the traversal.

    Examples
    --------
    >>> nodes = traverse(query="example", edges=[("keywords", "keywords")], ...)
    """
```

**Docstring Requirements**:
- ✅ All public functions and classes must have docstrings
- ✅ Use NumPy-style parameter descriptions
- ✅ Include type information in narrative (already in hints)
- ✅ Provide examples for complex functionality
- ❌ Tests and internal modules (`_base.py`) don't require docstrings

### 7.4 Naming Conventions

**Modules**:
- `lowercase_with_underscores` (public modules)
- `_leading_underscore.py` (private/internal modules)

**Classes**:
- `PascalCase` (e.g., `GraphRetriever`, `MetadataEdgeFunction`)

**Functions/Methods**:
- `snake_case` (e.g., `traverse`, `get_by_ids`)
- `async` functions prefixed with `a` (e.g., `atraverse`, `asearch`)

**Constants**:
- `UPPER_SNAKE_CASE` (e.g., `ALL_STORES`, `DEFAULT_K`)

**Private**:
- `_leading_underscore` (e.g., `_Traversal`, `_conversion.py`)

### 7.5 Code Patterns

#### Edge Definition Pattern

```python
# EdgeSpec: (source_field, target_field)
edges = [
    ("keywords", "keywords"),        # Bi-directional: match on same field
    ("mentions", "$id"),             # Outgoing: source field → document IDs
    ("$id", "related_ids"),          # Incoming: document ID → target field
]
```

#### Strategy Pattern

```python
# All strategies inherit from base Strategy class
class CustomStrategy(Strategy):
    def iteration(self, tracker: NodeTracker, depth: int) -> Iterable[Node]:
        """Select nodes at the current depth."""
        # Implementation
```

#### Adapter Pattern

```python
# All adapters inherit from base Adapter class
class CustomAdapter(Adapter):
    def search(self, query: str, k: int, filter: dict | None = None) -> list[Content]:
        """Perform similarity search."""
        # Implementation

    async def asearch(self, query: str, k: int, filter: dict | None = None) -> list[Content]:
        """Async similarity search."""
        # Implementation
```

#### Async Support

```python
# Always provide both sync and async versions
def traverse(...) -> list[Node]:
    """Sync version."""
    pass

async def atraverse(...) -> list[Node]:
    """Async version."""
    pass
```

#### Dataclasses for Models

```python
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)  # Immutable
class Node:
    id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any]
    depth: int
```

---

## 8. CI/CD Pipeline

### 8.1 GitHub Actions Workflows

**Main Workflow** (`.github/workflows/main.yml`):

**Triggers**:
- Push to `main` branch
- Pull requests (opened, synchronized, reopened, ready_for_review)

**Jobs**:

1. **Quality Checks** (runs on all PRs):
   - `fmt-check`: Code formatting (ruff format)
   - `lint-check`: Linting (ruff check)
   - `lock-check`: Lock file consistency (uv lock --locked)
   - `dep-check`: Dependency issues (deptry)
   - `nbstripout-check`: Notebook output stripped

2. **Tests and Type Check** (matrix: Python 3.10, 3.11, 3.12, 3.13):
   - **3.10, 3.11**: In-memory stores only
   - **3.12**: All stores (AstraDB, Cassandra, Chroma, OpenSearch)
   - **3.13**: In-memory, no extras
   - Type checking with mypy
   - Coverage reporting (3.12 only) → Coveralls

3. **Build Docs**:
   - Build documentation (mkdocs build --strict)
   - Test Jupyter notebooks (pytest docs --nbmake)

4. **Deploy Docs** (on push to main):
   - Deploy to GitHub Pages (gh-pages branch)

**Release Workflow** (`.github/workflows/on-release-main.yml`):
- Triggered on GitHub release publish
- Updates version in all `pyproject.toml` files
- Builds packages (uv build --all-packages)
- Publishes to PyPI with OIDC authentication
- Requires approval for `pypi` deployment environment

### 8.2 CI Environment Variables

**Required Secrets**:
- `ASTRA_DB_APPLICATION_TOKEN`: AstraDB authentication
- `ASTRA_DB_API_ENDPOINT`: AstraDB endpoint
- `OPENAI_API_KEY`: OpenAI API for documentation notebooks

**Dynamic Variables**:
- `ASTRA_DB_KEYSPACE`: `ci_${{ github.run_id }}_${{ strategy.job-index }}_${{ github.run_attempt }}`

### 8.3 Pre-Commit Checks

Before committing, ensure:

```bash
# 1. Format and lint
uv run poe fmt-fix
uv run poe lint-fix

# 2. Type check
uv run poe type-check

# 3. Run tests
uv run poe test

# 4. Strip notebook output
uv run poe nbstripout

# 5. Check lock file
uv run poe lock-check

# Or run everything:
uv run poe lint  # Runs all checks + docs build
```

---

## 9. Important Patterns and Anti-Patterns

### 9.1 DO ✅

**Code Organization**:
- ✅ Keep core logic in `graph-retriever` (no LangChain dependencies)
- ✅ Put LangChain-specific code in `langchain-graph-retriever`
- ✅ Use adapters for vector store abstractions
- ✅ Implement both sync and async versions of functions
- ✅ Use frozen dataclasses for immutable data structures

**Testing**:
- ✅ Test both sync and async code paths
- ✅ Use fixtures for reusable test data
- ✅ Mark optional dependency tests with `@pytest.mark.extra`
- ✅ Test edge cases (empty results, invalid inputs)
- ✅ Use testcontainers for external services

**Documentation**:
- ✅ Write NumPy-style docstrings for public APIs
- ✅ Include examples in docstrings
- ✅ Update documentation when changing behavior
- ✅ Use type hints to document parameter types

**Type Safety**:
- ✅ Use type hints for all function signatures
- ✅ Use `from __future__ import annotations` for forward references
- ✅ Run mypy before committing
- ✅ Use `typing-extensions` for compatibility

**Dependencies**:
- ✅ Keep dependencies minimal in core library
- ✅ Use optional dependencies for extras (transformers, stores)
- ✅ Pin versions in `uv.lock` (not in `pyproject.toml`)

### 9.2 DON'T ❌

**Code Organization**:
- ❌ Don't add LangChain dependencies to `graph-retriever`
- ❌ Don't duplicate logic between packages
- ❌ Don't bypass the adapter pattern for vector stores
- ❌ Don't use mutable default arguments (`def foo(bar=[])`)

**Testing**:
- ❌ Don't commit notebooks with output
- ❌ Don't hardcode credentials in tests
- ❌ Don't skip cleanup (use fixtures with `yield`)
- ❌ Don't test implementation details (test behavior)
- ❌ Don't create tests that depend on external state

**Documentation**:
- ❌ Don't leave public APIs without docstrings
- ❌ Don't use inconsistent docstring styles
- ❌ Don't forget to update docs when changing APIs
- ❌ Don't include implementation details in public docs

**Type Safety**:
- ❌ Don't use `Any` unless absolutely necessary
- ❌ Don't ignore mypy errors (fix them or use `# type: ignore` with comment)
- ❌ Don't use bare `except:` (specify exception types)

**Dependencies**:
- ❌ Don't add unnecessary dependencies
- ❌ Don't use deprecated APIs
- ❌ Don't modify `uv.lock` manually (use `uv run poe lock-fix`)

### 9.3 Common Pitfalls

**1. Edge Definition Confusion**

```python
# ❌ WRONG: Reversed source/target
edges = [("$id", "mentions")]  # This won't work as expected

# ✅ CORRECT: Links from "mentions" field to document IDs
edges = [("mentions", "$id")]
```

**2. Async/Sync Mixing**

```python
# ❌ WRONG: Calling async function without await
result = adapter.asearch(query, k=10)  # Returns coroutine, not result

# ✅ CORRECT: Use await
result = await adapter.asearch(query, k=10)
```

**3. Strategy Mutation**

```python
# ❌ WRONG: Strategies are mutated during traversal
strategy = Eager(select_k=10)
traverse(query, strategy=strategy, ...)  # Strategy is modified!
traverse(query, strategy=strategy, ...)  # Second call has stale state

# ✅ CORRECT: Strategy is deep-copied in traverse()
# This is handled automatically, but be aware
```

**4. Metadata Filter Format**

```python
# ❌ WRONG: Using store-specific filter format
metadata_filter = {"bool": {"must": [...]}}  # OpenSearch-specific

# ✅ CORRECT: Use generic format (adapter handles translation)
metadata_filter = {"keywords": "mammal"}
```

**5. Test Store Configuration**

```python
# ❌ WRONG: Testing all stores by default
pytest  # Runs against all stores (slow, requires credentials)

# ✅ CORRECT: Default to in-memory, opt-in to all stores
pytest              # In-memory only
pytest --stores=all # All stores (requires env vars)
```

---

## 10. Release Process

### 10.1 Version Management

**Version Locations** (must all match):
1. `packages/graph-retriever/pyproject.toml:2` → `version = "x.y.z"`
2. `packages/langchain-graph-retriever/pyproject.toml:2` → `version = "x.y.z"`

### 10.2 Release Steps

1. **Check draft release**:
   - Go to https://github.com/datastax/graph-rag/releases
   - Review suggested version number (auto-generated from commits)

2. **Create version bump PR**:
   ```bash
   git checkout -b release/v0.2.0
   # Edit packages/graph-retriever/pyproject.toml
   # Edit packages/langchain-graph-retriever/pyproject.toml
   git commit -am "Bump version to 0.2.0"
   git push origin release/v0.2.0
   # Create PR and merge to main
   ```

3. **Publish release**:
   - Edit draft release
   - Ensure version and tag match (e.g., `v0.2.0`)
   - Check "Set as a pre-release" (will be updated by automation)
   - Click "Publish release"

4. **Approve PyPI deployment**:
   - GitHub Actions will trigger
   - Approve `pypi` deployment environment for both packages
   - Automation publishes to PyPI

5. **Verify release**:
   ```bash
   pip install graph-retriever==0.2.0
   pip install langchain-graph-retriever==0.2.0
   ```

**Release Automation**:
- Version bumping (if not already done)
- Package building (`uv build --all-packages`)
- PyPI publishing with OIDC (no tokens needed)
- GitHub release notes generation

---

## 11. Tips for AI Assistants

### 11.1 Understanding User Intent

**Common Requests**:

| User Says | Likely Intent | Action |
|-----------|---------------|--------|
| "Add support for X vector store" | New adapter | Create adapter in `langchain-graph-retriever/adapters/` |
| "Improve traversal performance" | Optimization | Focus on `graph-retriever/traversal.py` and strategies |
| "Add keyword extraction" | Transformer | Create transformer in `langchain-graph-retriever/transformers/` |
| "Fix tests failing on X" | Bug fix | Check CI logs, run `uv run poe test-all` |
| "Update docs for X" | Documentation | Edit `docs/guide/` or update docstrings |

### 11.2 Code Navigation

**Finding Relevant Code**:

```bash
# Find where a class/function is defined
uv run rg "class GraphRetriever"
uv run rg "def traverse"

# Find usage examples
uv run rg "GraphRetriever\(" --type py

# Find tests for a module
uv run rg "test_.*traverse" --type py packages/*/tests/
```

**Key Entry Points**:
1. **Traversal**: `packages/graph-retriever/src/graph_retriever/traversal.py:14`
2. **LangChain Integration**: `packages/langchain-graph-retriever/src/langchain_graph_retriever/graph_retriever.py:24`
3. **Adapter Base**: `packages/graph-retriever/src/graph_retriever/adapters/base.py`
4. **Strategy Base**: `packages/graph-retriever/src/graph_retriever/strategies/base.py`

### 11.3 Testing Your Changes

**Minimal Testing** (fast):
```bash
uv run poe test-gr        # Core library only
uv run poe fmt-fix        # Auto-format
uv run poe lint-check     # Check lints
```

**Full Testing** (before PR):
```bash
uv run poe lint           # All checks
uv run poe test-all       # All stores (requires env vars)
uv run poe test-nb        # Notebooks
```

**Debugging Tests**:
```bash
# Run specific test file
uv run pytest packages/graph-retriever/tests/test_traversal.py -v

# Run specific test
uv run pytest packages/graph-retriever/tests/test_traversal.py::test_basic_traversal -v

# Run with prints
uv run pytest packages/graph-retriever/tests/test_traversal.py -v -s

# Run with specific store
uv run pytest packages/langchain-graph-retriever/tests/ --stores=chroma -v
```

### 11.4 Common Modifications

#### Adding a New Vector Store Adapter

1. Create adapter in `packages/langchain-graph-retriever/src/langchain_graph_retriever/adapters/`:
   ```python
   from graph_retriever.adapters import Adapter
   from langchain_core.vectorstores import VectorStore

   class NewStoreAdapter(Adapter):
       def __init__(self, store: VectorStore):
           self.store = store

       def search(self, query: str, k: int, filter: dict | None = None) -> list[Content]:
           # Implementation
           pass

       async def asearch(self, query: str, k: int, filter: dict | None = None) -> list[Content]:
           # Async implementation
           pass
   ```

2. Register in `adapters/inference.py`:
   ```python
   ADAPTER_MAP = {
       "NewStoreVectorStore": NewStoreAdapter,
       # ...
   }
   ```

3. Add tests in `packages/langchain-graph-retriever/tests/adapters/test_new_store.py`:
   ```python
   class TestNewStoreAdapter(AdapterComplianceSuite):
       # Inherit standard tests
       pass
   ```

#### Adding a New Traversal Strategy

1. Create strategy in `packages/graph-retriever/src/graph_retriever/strategies/`:
   ```python
   from graph_retriever.strategies.base import Strategy, NodeTracker

   class CustomStrategy(Strategy):
       def iteration(self, tracker: NodeTracker, depth: int) -> Iterable[Node]:
           # Select nodes at current depth
           pass
   ```

2. Add tests in `packages/graph-retriever/tests/strategies/test_custom.py`

3. Document in `docs/guide/strategies.md`

#### Adding a New Document Transformer

1. Create transformer in `packages/langchain-graph-retriever/src/langchain_graph_retriever/transformers/`:
   ```python
   from langchain_core.documents import Document
   from langchain_core.documents.transformers import BaseDocumentTransformer

   class CustomTransformer(BaseDocumentTransformer):
       def transform_documents(self, documents: list[Document]) -> list[Document]:
           # Transform documents
           pass
   ```

2. Mark optional dependency in `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   custom = ["custom-library>=1.0.0"]
   ```

3. Add test with `@pytest.mark.extra`:
   ```python
   @pytest.mark.extra
   def test_custom_transformer():
       # Test implementation
       pass
   ```

### 11.5 Documentation Updates

**When to Update Docs**:
- ✅ New features or APIs
- ✅ Behavior changes
- ✅ Breaking changes
- ✅ New examples or use cases

**Where to Update**:
- **API changes**: Update docstrings → auto-generates `docs/reference/`
- **User guides**: Edit `docs/guide/*.md`
- **Examples**: Add/update notebooks in `docs/examples/`
- **Blog posts**: Create in `docs/blog/posts/`

**Preview Changes**:
```bash
uv run poe docs-serve  # Opens browser with live reload
```

### 11.6 Debugging Tips

**Enable Debug Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Inspect Nodes**:
```python
nodes = traverse(query, ...)
for node in nodes:
    print(f"ID: {node.id}, Depth: {node.depth}, Score: {node.score}")
    print(f"Edges: {node.edges}")
```

**Test with Small Dataset**:
```python
from graph_rag_example_helpers.datasets import animals
docs = animals.fetch()[:5]  # Use only 5 documents
```

**Use In-Memory Store**:
```python
from langchain_graph_retriever.adapters import InMemoryAdapter
# Faster than external stores, easier to debug
```

---

## 12. Quick Reference

### 12.1 Essential Commands

```bash
# Setup
uv sync                    # Install dependencies
uv run poe sync           # Install all packages + extras

# Development
uv run poe fmt-fix        # Auto-format code
uv run poe lint-fix       # Auto-fix lints
uv run poe type-check     # Type checking
uv run poe lint           # All checks + docs build

# Testing
uv run poe test           # Unit tests (in-memory)
uv run poe test-all       # All tests (all stores)
uv run poe test-nb        # Notebook tests
uv run poe coverage       # Coverage report

# Documentation
uv run poe docs-serve     # Live preview
uv run poe docs-build     # Build docs

# Notebooks
uv run poe nbstripout     # Strip output
```

### 12.2 File Locations

| What | Where |
|------|-------|
| Core traversal | `packages/graph-retriever/src/graph_retriever/traversal.py` |
| GraphRetriever | `packages/langchain-graph-retriever/src/langchain_graph_retriever/graph_retriever.py` |
| Adapters | `packages/langchain-graph-retriever/src/langchain_graph_retriever/adapters/` |
| Strategies | `packages/graph-retriever/src/graph_retriever/strategies/` |
| Transformers | `packages/langchain-graph-retriever/src/langchain_graph_retriever/transformers/` |
| Tests | `packages/*/tests/` |
| Docs | `docs/` |
| Examples | `docs/examples/` |
| Datasets | `data/` |

### 12.3 Key Concepts

**Edge Spec**: `(source_field, target_field)` tuple defining graph relationships

**Node**: Document with `id`, `content`, `embedding`, `metadata`, `depth`, `edges`, `score`

**Strategy**: Defines how nodes are selected during traversal (Eager, MMR, Scored)

**Adapter**: Interface to vector stores (search, get_by_ids)

**Transformer**: LangChain document processor (NER, keywords, HTML, etc.)

**Traversal**: Vector search → extract edges → traverse graph → select nodes

---

## 13. Resources

### Documentation
- **Main Docs**: https://datastax.github.io/graph-rag
- **GitHub**: https://github.com/datastax/graph-rag
- **Issues**: https://github.com/datastax/graph-rag/issues

### Tools
- **UV**: https://docs.astral.sh/uv/
- **Ruff**: https://docs.astral.sh/ruff/
- **Mypy**: https://mypy.readthedocs.io/
- **Pytest**: https://docs.pytest.org/

### LangChain
- **LangChain Docs**: https://python.langchain.com/docs/
- **Vector Stores**: https://python.langchain.com/docs/integrations/vectorstores/
- **Retrievers**: https://python.langchain.com/docs/modules/data_connection/retrievers/

---

## 14. Change Log

| Date | Changes |
|------|---------|
| 2025-11-15 | Initial CLAUDE.md creation |

---

**Questions or Issues?**

If you encounter problems or have questions:
1. Check the [documentation](https://datastax.github.io/graph-rag)
2. Search [existing issues](https://github.com/datastax/graph-rag/issues)
3. Create a new issue with detailed information
4. Ask in pull request comments

**Happy Coding! 🚀**
