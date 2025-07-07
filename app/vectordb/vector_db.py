
# app/vectordb/vector_db.py
import os
import threading
from functools import lru_cache
from typing import List

import chromadb
from app.cache.cache_db import get_cache_db
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from datetime import datetime, timedelta

# ───────── 설정 상수 ───────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")  # 도커 외부 접근 시
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

class VectorDB:
    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings()
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

    def get_docs(self, file_id: str, k: int = 30) -> List[Document]:
        try:
            collection_name = self._get_collection_name(file_id)
            vectorstore = self._get_vectorstore(collection_name)
            return vectorstore.similarity_search("summary", k=k)  # 빈 query 대신 dummy query
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

    def cleanup_unused_vectors(self, cache) -> List[str]:
        deleted = []
        try:
            vector_ids = self.list_stored_documents()
            for fid in vector_ids:
                if not cache.get_pdf(fid):
                    self.delete_document(fid)
                    self.log_vector_deletion(fid)
                    deleted.append(fid)
        except Exception as e:
            print(f"[VectorDB Cleanup] Error: {e}")
        return deleted

    def log_vector_deletion(self, file_id: str):
        now = datetime.now()
        date_key = f"vector:deleted:{now.strftime('%Y-%m-%d')}"
        self.r = get_cache_db().r  # Redis 인스턴스 (불러오거나 주입 가능)
        self.r.rpush(date_key, f"{file_id}|{now.isoformat()}")
@lru_cache(maxsize=1)
def get_vector_db() -> "VectorDB":
    return VectorDB()

