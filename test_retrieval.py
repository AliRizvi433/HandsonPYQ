
from pyqproj.document_loader import DocumentLoader
from pyqproj.vector_store import PolicyVectorStore


# Step 1: Load documents
loader = DocumentLoader()
documents = loader.create_documents()

print(f"Loaded {len(documents)} documents")

# Step 2: Create FAISS index
store = PolicyVectorStore()
store.create_vector_store(documents)

print("FAISS index created successfully!")

# Step 3: Search documents
questions = [
    "How many sick leave days do employees get?",
    "Can I work from home?",
    "What is the annual leave policy?"
]

for question in questions:
    print(f"\nQUESTION: {question}")

    results = store.similarity_search_sync(
        question,
        k=3
    )

    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Type:", doc.metadata.get("type"))
        print(
            "Title:",
            doc.metadata.get(
                "title",
                doc.metadata.get("name")
            )
        )
        print("Content:", doc.page_content[:200])

# Step 4: Save index
store.save_local("faiss_index")
print("\nFAISS index saved successfully!")

# Step 5: Load saved index
loaded_store = PolicyVectorStore.load_local(
    "faiss_index"
)

print("FAISS index loaded successfully!")

# Step 6: Search loaded index
results = loaded_store.similarity_search_sync(
    "What is the sick leave policy?",
    k=2
)

print("\nResults from loaded index:")

for doc in results:
    print(
        doc.metadata.get(
            "title",
            doc.metadata.get("name")
        )
    )
