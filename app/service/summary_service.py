
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from app.vectordb.vector_db import VectorDB
from app.cache.cache_db import get_cache_db  # ✅ Redis 기반 캐시
from app.cache.cache_db import RedisCacheDB  # ✅ 명시적으로 타입 지정

class SummaryService:
    def __init__(self, vector: VectorDB, cache: RedisCacheDB = get_cache_db()):
        self.vector, self.cache = vector, cache
        self.llm = ChatOpenAI(temperature=0.3)

    def generate(self, file_id: str) -> str:
        # ✅ Redis 캐시에서 먼저 요약 확인
        if (c := self.cache.get_pdf(file_id)):
            return c
        
        # ✅ 벡터 DB에서 문서 가져오기
        docs = self.vector.get_docs(file_id)
        if not docs:
            return f"No documents found for file_id='{file_id}'."
        
        # ✅ LangChain map-reduce 요약 체인 사용
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        summary = chain.run(docs)
        
        # ✅ Redis에 캐시 저장
        self.cache.set_pdf(file_id, summary)
        return summary

