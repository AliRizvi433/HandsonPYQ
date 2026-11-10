
import pytest

from pyqproj.vector_store import PolicyVectorStore
from pyqproj.rag_chain import HRPolicyRAG


@pytest.fixture
def rag():

    store = PolicyVectorStore()

    return HRPolicyRAG(
        vector_store=store,
        hris_data_path="data/hris_api.json"
    )


@pytest.mark.asyncio
async def test_valid_employee(rag):

    result = await rag.call_hris_api("EMP101")

    assert result["name"] == "Emma Wilson"
    assert result["pto_balance"] == 8


@pytest.mark.asyncio
async def test_invalid_employee(rag):

    result = await rag.call_hris_api("EMP999")

    assert result == {
        "error": "Employee ID EMP999 not found"
    }


@pytest.mark.asyncio
async def test_numeric_employee_id(rag):

    result = await rag.call_hris_api(12345)

    assert result == {
        "error": "Employee ID 12345 not found"
    }


def test_hris_dictionary(rag):

    assert len(rag.hris_data) == 6
    assert "EMP101" in rag.hris_data


def test_chain_created(rag):

    assert rag.chain is not None


@pytest.mark.asyncio
async def test_personal_query_without_employee_id(rag):

    response = await rag.query(
        "What is my PTO balance?"
    )

    assert "employee ID" in response


@pytest.mark.asyncio
async def test_vacation_query_without_employee_id(rag):

    response = await rag.query(
        "How many vacation days do I have left?"
    )

    assert "employee ID" in response


@pytest.mark.asyncio
async def test_personal_query_invalid_employee(rag):

    response = await rag.query(
        "What is my PTO balance?",
        employee_id="EMP999"
    )

    assert response == "Employee ID EMP999 not found"
