# app/vectordb/vector_db.py
import os
import threading
from functools import lru_cache
from typing import List

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

# ───────── 설정 상수 ───────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")  # 도커 외부 접근 시
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "9000"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

def get_embedding_model():
    if LLM_PROVIDER == "hf":
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, 
                                     model_kwargs={"device": "cpu"},
                                     encode_kwargs={"normalize_embeddings": True})
    else:
        return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

class VectorDB:
    def __init__(self) -> None:
        #self.embeddings = OpenAIEmbeddings()
        self.embeddings = get_embedding_model()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self._lock = threading.RLock()

        self.client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    def _get_collection_name(self, file_id: str) -> str:
        return file_id

    def _get_vectorstore(self, collection_name: str) -> Chroma:
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

    def store(self, text: str, file_id: str) -> None:
        with self._lock:
            collection_name = self._get_collection_name(file_id)
            vectorstore = self._get_vectorstore(collection_name)

            chunks = self.text_splitter.split_text(text)
            documents = [
                Document(
                    page_content=chunk,
                    metadata={"file_id": file_id, "chunk_index": i}
                )
                for i, chunk in enumerate(chunks)
            ]

            vectorstore.add_documents(documents)

    def get_docs(self, file_id: str) -> List[Document]:
        try:
            collection_name = self._get_collection_name(file_id)
            vectorstore = self._get_vectorstore(collection_name)
            return vectorstore.similarity_search("요약", k=100)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def delete_document(self, file_id: str) -> bool:
        try:
            with self._lock:
                collection_name = self._get_collection_name(file_id)
                self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False

    def list_stored_documents(self) -> List[str]:
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []


@lru_cache(maxsize=1)
def get_vector_db() -> "VectorDB":
    return VectorDB()

