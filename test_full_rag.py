
import asyncio

from pyqproj.vector_store import PolicyVectorStore
from pyqproj.rag_chain import HRPolicyRAG


async def main():

    # Load the FAISS index created earlier
    store = PolicyVectorStore.load_local(
        "faiss_index"
    )

    # Initialize the RAG application
    rag = HRPolicyRAG(
        vector_store=store,
        hris_data_path="data/hris_api.json"
    )

    # Test 1: General policy question
    print("\nTEST 1: General Policy")
    print("-" * 50)

    response = await rag.query(
        "How many sick leave days do employees get?"
    )

    print(response)

    # Test 2: Work from home
    print("\nTEST 2: Work From Home")
    print("-" * 50)

    response = await rag.query(
        "Can employees work from home?"
    )

    print(response)

    # Test 3: Personal PTO balance
    print("\nTEST 3: Personal HRIS")
    print("-" * 50)

    response = await rag.query(
        "What is my PTO balance?",
        employee_id="EMP101"
    )

    print(response)

    # Test 4: Invalid employee
    print("\nTEST 4: Invalid Employee")
    print("-" * 50)

    response = await rag.query(
        "What is my PTO balance?",
        employee_id="EMP999"
    )

    print(response)


if __name__ == "__main__":
    asyncio.run(main())
