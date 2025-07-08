from langchain_openai import ChatOpenAI
from app.vectordb.vector_db import VectorDB
from app.cache.cache_db import get_cache_db, RedisCacheDB
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator

# ───────── State 정의 ───────────────────────────────
class RAGState(TypedDict):
    file_id: str
    query: str
    top_k: int
    docs: List
    context: str
    retrieve_score: str
    summary: str
    summary_score: str
    translated: str
    retry_count: int
    max_retries: int

class LangchainRAGPipeline:
    def __init__(self, vector: VectorDB, cache: RedisCacheDB = get_cache_db(), user_lang: str = "ko"):
        self.vector = vector
        self.cache = cache
        self.user_lang = user_lang
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        
        # 단순한 요약 프롬프트들
        self.rag_prompt = PromptTemplate(
            template="다음 문서 내용을 요약해주세요:\n{text}\n\n요약:",
            input_variables=["text"]
        )
        
        self.rag_completeness_prompt = PromptTemplate(
            template="다음 요약 내용의 완성도를 0-1로 평가해주세요.\n요약: {text}\n기준: 문서의 핵심 내용이 명확하고 완전하게 포함되어야 함\n점수:",
            input_variables=["text"]
        )
        
        self.summary_prompt = PromptTemplate(
            template="다음 내용을 더 간결하게 요약해주세요:\n{text}\n\n요약:",
            input_variables=["text"]
        )
        
        self.summary_completeness_prompt = PromptTemplate(
            template="다음 요약 내용의 완성도를 0-1로 평가해주세요.\n요약: {text}\n기준: 핵심 내용이 간결하면서도 완전하게 포함되어야 함\n점수:",
            input_variables=["text"]
        )
        
        self.translate_prompt = PromptTemplate(
            template="다음 텍스트를 {lang}로 자연스럽게 번역해주세요:\n{text}\n\n번역:",
            input_variables=["text", "lang"]
        )
        
        # LangGraph 워크플로우 생성
        self.workflow = self._create_workflow()
    
    def _create_workflow(self):
        """
        LangGraph 워크플로우 생성
        
        Graph 구조 요약:
            Retrieve → Retrieve 완성도 확인 → Generate → Generate 완성도 확인 → 번역
            각 단계에서 완성도가 낮으면 이전 노드로 이동하여 재시도
        """

        workflow = StateGraph(RAGState)

        # 노드 추가
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("retrieve_completeness", self._retrieve_completeness)
        workflow.add_node("generate", self._generate)
        workflow.add_node("generate_completeness", self._generate_completeness)
        workflow.add_node("translate", self._translate)

        # 순차 엣지 추가
        workflow.add_edge("retrieve", "retrieve_completeness")
        workflow.add_edge("generate", "generate_completeness")
        workflow.add_edge("translate", END)
        
        # 조건부 엣지 추가
        workflow.add_conditional_edges(
            "retrieve_completeness",
            self._should_retry_retrieve,
            {
                "retry_retrieve": "retrieve",
                "continue": "generate",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "generate_completeness",
            self._should_retry_generate,
            {
                "retry_generate": "generate",
                "continue": "translate"
            }
        )
        
        return workflow.compile()
    
    def _retrieve(self, state: RAGState) -> RAGState:
        """Retrieve 노드 - 유사도 검색만 수행"""
        # 1. Retrieval: 관련 문서 검색
        docs = self.vector.similarity_search(state["query"], file_id=state["file_id"], k=state["top_k"])
        if not docs:
            state["docs"] = []
            state["context"] = ""
            state["summary"] = "검색된 문서가 없습니다."
            state["translated"] = "검색된 문서가 없습니다."
            return state
        
        # 2. Context 생성
        context = "\n\n".join([d.page_content for d in docs])
        
        state["docs"] = docs
        state["context"] = context
        
        return state
    
    def _retrieve_completeness(self, state: RAGState) -> RAGState:
        """Retrieve 완성도 검증 노드"""
        if not state["docs"]:
            state["retrieve_score"] = "0.0"
            return state
            
        # 검색된 문서의 관련성 평가
        retrieve_completeness_prompt = PromptTemplate(
            template="다음 검색된 문서들이 질문과 관련성이 있는지 0-1로 평가해주세요.\n\n질문: {query}\n\n검색된 문서들:\n{context}\n\n관련성 점수 (0-1):",
            input_variables=["query", "context"]
        )
        
        prompt = retrieve_completeness_prompt.format(
            query=state["query"], 
            context=state["context"][:1000]  # 처음 1000자만
        )
        result = self.llm.invoke(prompt)
        state["retrieve_score"] = result.content.strip()
        return state
    
    def _generate(self, state: RAGState) -> RAGState:
        """Generate 노드 - 요약문 생성"""
        # 요약문 생성
        prompt = self.rag_prompt.format(text=state["context"])
        result = self.llm.invoke(prompt)
        state["summary"] = result.content.strip()
        return state
    
    def _generate_completeness(self, state: RAGState) -> RAGState:
        """Generate 완성도 검증 노드"""
        prompt = self.summary_completeness_prompt.format(text=state["summary"])
        result = self.llm.invoke(prompt)
        state["summary_score"] = result.content.strip()
        return state
    
    def _translate(self, state: RAGState) -> RAGState:
        """번역 노드"""
        prompt = self.translate_prompt.format(text=state["summary"], lang=self.user_lang)
        result = self.llm.invoke(prompt)
        state["translated"] = result.content.strip()
        return state
    
    def _should_retry_retrieve(self, state: RAGState) -> str:
        """Retrieve 재시도 여부 결정"""
        # 문서가 없는 경우 즉시 종료
        if not state["docs"]:
            return "end"
            
        score = self._score_to_float(state["retrieve_score"])
        if score < 0.7 and state["retry_count"] < state["max_retries"]:
            state["retry_count"] += 1
            return "retry_retrieve"
        return "continue"
    
    def _should_retry_generate(self, state: RAGState) -> str:
        """Generate 재시도 여부 결정"""
        score = self._score_to_float(state["summary_score"])
        if score < 0.7 and state["retry_count"] < state["max_retries"]:
            state["retry_count"] += 1
            return "retry_generate"
        return "continue"
    
    def _score_to_float(self, score_str: str) -> float:
        """점수 문자열을 float로 변환"""
        import re
        match = re.search(r"([01](?:\.\d+)?)", score_str)
        if match:
            return float(match.group(1))
        return 0.0
    
    def run(self, file_id: str, query: str = "이 문서의 주요 내용을 요약해줘", top_k: int = 5) -> str:
        """LangGraph 워크플로우 실행"""
        initial_state = RAGState(
            file_id=file_id,
            query=query,
            top_k=top_k,
            docs=[],
            context="",
            retrieve_score="",
            summary="",
            summary_score="",
            translated="",
            retry_count=0,
            max_retries=2
        )

        result = self.workflow.invoke(initial_state)
        return result["translated"]
    