# app/vectordb/vector_db.py
import os
import threading
from functools import lru_cache
from typing import List

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# tiktoken을 사용한 정확한 토큰 계산 (선택사항)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# ───────── 설정 상수 ───────────────────────────────
CHUNK_SIZE = 1000  # 토큰 기준으로 변경
CHUNK_OVERLAP = 200  # 토큰 기준으로 변경

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")  # 도커 외부 접근 시
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

class VectorDB:
    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=self._token_length,  # 토큰 기반 길이 계산
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],  # 한국어에 적합한 구분자
        )
        self._lock = threading.RLock()

        self.client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    def _token_length(self, text: str) -> int:
        """텍스트의 토큰 수를 계산"""
        if TIKTOKEN_AVAILABLE:
            # tiktoken을 사용한 정확한 토큰 계산
            try:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                return len(encoding.encode(text))
            except:
                pass
        
        # fallback: 대략적인 토큰 수 계산 (한국어 고려)
        # 한국어는 평균적으로 1글자 = 1토큰, 영어는 1단어 = 1.3토큰 정도
        korean_chars = sum(1 for char in text if '\u3131' <= char <= '\u318E' or '\uAC00' <= char <= '\uD7A3')
        english_words = len([word for word in text.split() if word.isascii()])
        
        # 한국어 글자 + 영어 단어 * 1.3
        return korean_chars + int(english_words * 1.3)

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
        """file_id로 저장된 모든 청크를 LangChain Document 배열로 반환."""
        raw = self.client.get_collection(self._get_collection_name(file_id)).get(include=["documents"], where={"file_id": file_id})
        docs = raw.get("documents", [])

        # langchain-chroma 0.1.x REST 모드 → 문자열 리스트, 임베디드 모드 → Document 객체
        if docs and isinstance(docs[0], Document):
            return docs
        return [Document(page_content=t) for t in docs]

    def similarity_search(self, query: str, file_id: str = None, k: int = 5) -> List[Document]:
        """쿼리와 가장 유사한 상위 k개 청크를 반환."""
        return self.client.get_collection(self._get_collection_name(file_id)).similarity_search(
            query=query,
            k=k,
            where={"file_id": file_id}
        )
        

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

