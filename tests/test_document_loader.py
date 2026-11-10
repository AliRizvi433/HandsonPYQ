
from pyqproj.document_loader import DocumentLoader


def test_document_count():
    loader = DocumentLoader()
    documents = loader.create_documents()

    assert len(documents) == 16


def test_policy_documents():
    loader = DocumentLoader()
    documents = loader.create_documents()

    policies = [
        doc for doc in documents
        if doc.metadata["type"] == "policy"
    ]

    assert len(policies) == 10
    assert policies[0].metadata["policy_id"] == "POL001"
    assert "Sick Leave Policy" in policies[0].page_content


def test_employee_documents():
    loader = DocumentLoader()
    documents = loader.create_documents()

    employees = [
        doc for doc in documents
        if doc.metadata["type"] == "employee"
    ]

    assert len(employees) == 6
    assert employees[0].metadata["employee_id"] == "EMP101"
    assert "8 days" in employees[0].page_content
