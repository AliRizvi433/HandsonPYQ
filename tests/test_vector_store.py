
import pytest

from langchain_core.documents import Document
from pyqproj.document_loader import DocumentLoader
from pyqproj.vector_store import PolicyVectorStore


def test_empty_documents():
    store = PolicyVectorStore()

    with pytest.raises(ValueError):
        store.create_vector_store([])


def test_uninitialized_search():
    store = PolicyVectorStore()

    with pytest.raises(ValueError):
        store.similarity_search_sync("sick leave")


def test_duplicate_removal():
    store = PolicyVectorStore()

    documents = [
        Document(
            page_content="First policy",
            metadata={
                "type": "policy",
                "policy_id": "POL001",
                "title": "Sick Leave"
            }
        ),
        Document(
            page_content="Duplicate policy",
            metadata={
                "type": "policy",
                "policy_id": "POL001",
                "title": "Sick Leave"
            }
        ),
        Document(
            page_content="Second policy",
            metadata={
                "type": "policy",
                "policy_id": "POL002",
                "title": "Annual Leave"
            }
        )
    ]

    result = store._remove_duplicates(documents, 2)

    assert len(result) == 2


def test_vector_store_creation():
    loader = DocumentLoader()
    documents = loader.create_documents()

    store = PolicyVectorStore()
    store.create_vector_store(documents)

    assert store.vector_store is not None
    assert store.vector_store.index.ntotal == 16


def test_duplicate_titles():

    store = PolicyVectorStore()

    documents = [
        Document(
            page_content="First version",
            metadata={
                "type": "policy",
                "policy_id": "POL001",
                "title": "Sick Leave Policy"
            }
        ),
        Document(
            page_content="Second version",
            metadata={
                "type": "policy",
                "policy_id": "POL999",
                "title": "Sick Leave Policy"
            }
        ),
        Document(
            page_content="Annual leave information",
            metadata={
                "type": "policy",
                "policy_id": "POL002",
                "title": "Annual Leave Policy"
            }
        )
    ]

    result = store._remove_duplicates(documents, k=3)

    assert len(result) == 2
    assert result[0].metadata["title"] == "Sick Leave Policy"
    assert result[1].metadata["title"] == "Annual Leave Policy"


def test_zero_k():

    store = PolicyVectorStore()

    store.create_vector_store([
        Document(
            page_content="Test policy",
            metadata={"title": "Test"}
        )
    ])

    result = store.similarity_search_sync(
        "Test policy",
        k=0
    )

    assert result == []


def test_save_without_initialization(tmp_path):

    store = PolicyVectorStore()

    with pytest.raises(ValueError):
        store.save_local(str(tmp_path / "index"))


def test_load_missing_directory(tmp_path):

    missing_path = tmp_path / "nonexistent"

    with pytest.raises(FileNotFoundError):
        PolicyVectorStore.load_local(
            str(missing_path)
        )


@pytest.mark.asyncio
async def test_async_similarity_search():

    store = PolicyVectorStore()

    documents = [
        Document(
            page_content="Employees receive 12 sick leave days.",
            metadata={
                "type": "policy",
                "title": "Sick Leave Policy",
                "policy_id": "POL001"
            }
        ),
        Document(
            page_content="Employees may work from home.",
            metadata={
                "type": "policy",
                "title": "Remote Work Policy",
                "policy_id": "POL002"
            }
        )
    ]

    store.create_vector_store(documents)

    results = await store.similarity_search(
        "How many sick leave days do employees get?",
        k=2
    )

    assert len(results) == 2

    assert all(
        isinstance(doc, Document)
        for doc in results
    )
