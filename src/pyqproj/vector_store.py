
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class PolicyVectorStore:

    def __init__(self):
        load_dotenv()

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        self.vector_store = None

    def create_vector_store(
        self, documents: List[Document]
    ) -> None:

        if not documents:
            raise ValueError("No documents provided")

        self.vector_store = FAISS.from_documents(
            documents,
            self.embeddings
        )

    def _remove_duplicates(
        self, documents: List[Document], k: int
    ) -> List[Document]:

        if k <= 0:
            return []

        unique_documents = []
        seen = set()

        for doc in documents:

            metadata = doc.metadata

            if metadata.get("type") == "employee":
                identifier = (
                    "employee",
                    str(
                        metadata.get("employee_id")
                        or metadata.get("name")
                        or doc.page_content
                    )
                )

            else:
                identifier = (
                    "policy",
                    metadata.get("title")
                    or metadata.get("policy_id")
                    or doc.page_content
                )

            if identifier in seen:
                continue

            seen.add(identifier)
            unique_documents.append(doc)

            if len(unique_documents) >= k:
                break

        return unique_documents

    async def similarity_search(
        self, query: str, k: int = 5
    ) -> List[Document]:

        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        if k <= 0:
            return []

        total = self.vector_store.index.ntotal

        documents = await self.vector_store.asimilarity_search(
            query,
            k=total
        )

        return self._remove_duplicates(documents, k)

    def similarity_search_sync(
        self, query: str, k: int = 5
    ) -> List[Document]:

        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        if k <= 0:
            return []

        total = self.vector_store.index.ntotal

        documents = self.vector_store.similarity_search(
            query,
            k=total
        )

        return self._remove_duplicates(documents, k)

    def save_local(self, path: str) -> None:

        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        self.vector_store.save_local(path)

    @classmethod
    def load_local(cls, path: str):

        if not Path(path).is_dir():
            raise FileNotFoundError(
                f"Directory not found: {path}"
            )

        instance = cls()

        instance.vector_store = FAISS.load_local(
            path,
            instance.embeddings,
            allow_dangerous_deserialization=True
        )

        return instance
