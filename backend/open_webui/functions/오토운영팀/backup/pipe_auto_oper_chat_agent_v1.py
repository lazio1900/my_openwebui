############################################
# 작성자 : 이민재 / Claude
# 생성일자 : 2026-03-04
# 이력
# 2026-03-04 v1 : FAQ + 운영기준 통합 에이전트
#   - FAQ 챗봇(v11)과 운영기준 챗봇(v7)을 단일 Pipe로 통합
#   - FAQ wiki sync: init 시 Outline API에서 FAQ 다운로드 → ChromaDB 저장
#   - FAQ 충분성 판단: 단일 LLM 호출 (기존 grade_documents 개별 N회 → 1회)
#   - kure 임베딩 통일 (FAQ도 kure 사용, 기존 FAQ의 koe5에서 변경)
#   - ENABLE_FAQ_SEARCH=False 시 v7 동일 동작 (하위 호환)
#   - ENABLE_FAQ_WIKI_SYNC=False 시 기존 ChromaDB 데이터 사용 (빠른 init)
#   - 출처(citation) HTML 카드형 포맷 (_format_citation_html)
#
# ──────────────────────────────────────────
# 아키텍처 요약
# ──────────────────────────────────────────
# [흐름] FAQ 우선 → 운영기준 폴백
#
#   사용자 질문
#       ↓
#   [1] _faq_search()      : FAQ 벡터 검색 (LLM 호출 없음)
#       ↓
#   [2] _evaluate_faq()    : FAQ 충분성 판단 (LLM 1회)
#       ↓
#       ├─ 충분 → _generate(faq)  → 종료   ← classify 스킵
#       ↓
#   [3] _classify()        : 도메인/서브태스크 분류 (LLM 1회)
#       ↓
#       ├─ 재질의 → _requery()    → 종료
#       ↓
#   [4] _react_agent()     : 운영기준 ReAct 검색 (LLM 1~2회)
#       ↓
#   [5] _generate(standard) → 종료
#
# [핵심 메서드]
#   pipe()                  : 메인 엔트리포인트 (위 흐름 제어)
#   _faq_search()           : FAQ 컬렉션 벡터 검색 (FAQ_EMBEDDING_K개)
#   _evaluate_faq()         : FAQ 충분성 LLM 판단 (sufficient: true/false)
#   _classify()             : 도메인/서브태스크 분류 + 재질의 감지
#   _react_agent()          : ReAct 루프 (Think→Act→Observe, 최대 3회)
#   _react_step()           : ReAct 단일 스텝 LLM 호출
#   _execute_tool()         : search_documents 도구 실행
#   _rrf_merge()            : 벡터+BM25 하이브리드 검색 RRF 결합
#   _generate()             : 최종 답변 생성 (answer_source별 프롬프트 분기)
#   _generate_with_image()  : 이미지 포함 답변 (운영기준 전용)
#   _generate_followups()   : 후속질문 생성 (병렬)
#   _check_grounding()      : 답변 신뢰도 검증 (병렬)
#   _format_citation_html() : 출처를 HTML 카드로 포맷팅
#   _requery()              : 재질의 응답
#   _proc_wiki_faq()        : Outline wiki → Document 리스트
#   _save_to_chroma()       : Document → ChromaDB 저장
#
# [출처 표시]
#   FAQ  : "#### [FAQ]"              (서브태스크 없음)
#   표준 : "#### [도메인 > 서브태스크]"  (예: 중고승용 운영기준 > (론/할부))
#
# [citation 포맷]
#   FAQ  : HTML 카드 (Q. 질문 박스 + 답변 + 원문 링크)
#   표준 : page_content를 markdown→HTML 변환
#
# [소스 파일]
#   v7.py     : 운영기준 챗봇 (Orchestrator + ReAct)
#   FAQ v11   : FAQ 챗봇 (wiki sync, 이미지 처리)
#   v9_backup : FAQ 분기 패턴 참조
############################################

##########################################################################
# 라이브러리 임포트
##########################################################################
from pydantic import BaseModel
from typing import Dict, List, Callable, Awaitable, Optional, Protocol
import re
import time
import json
import asyncio
import sys
import os
import requests
import random

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from markdown import markdown
from urllib.parse import urlparse, parse_qs

from open_webui.utils.misc import get_last_user_message
from open_webui.models.users import Users
from open_webui.utils.chat import generate_chat_completion
from fastapi import Request


##########################################################################
# 비동기 이벤트 통신 유틸리티
##########################################################################
EmitterType = Optional[Callable[[dict], Awaitable[None]]]


class SendStatusType(Protocol):
    def __call__(self, status_message: str, done: bool) -> Awaitable[None]: ...


class SendCitationType(Protocol):
    def __call__(self, url: str, title: str, content: str) -> Awaitable[None]: ...


def get_send_status(__event_emitter__: EmitterType):
    async def send_status(status_message: str, done: bool):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {"type": "status", "data": {"description": status_message, "done": done}}
        )

    return send_status


def get_send_citation(__event_emitter__: EmitterType):
    async def send_citation(url: str, title: str, content: str):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": [content],
                    "metadata": [{"source": url, "html": True}],
                    "source": {"name": title},
                },
            }
        )

    return send_citation


##########################################################################
# 도메인/서브태스크 매핑
##########################################################################

DOMAIN_SUBTASK_MAP = {
    "중고승용": {
        "subtasks": ["(론/할부)", "(임직원대출)", "(신용구제)", "(Dual Offer)", "(엔카)"],
        "display_name": "중고승용 운영기준",
        "has_image": True,
    },
    "전략금융": {
        "subtasks": [
            "(재고금융)",
            "(제휴점 운영자금)",
            "(매매상사 운영자금)",
            "(운영자금 자금용도 기준)",
            "(임차보증금)",
        ],
        "display_name": "전략금융 운영기준",
        "has_image": False,
    },
    "중고리스": {
        "subtasks": ["(중고리스)"],
        "display_name": "중고리스 운영기준",
        "has_image": False,
    },
    "중형트럭": {
        "subtasks": ["(중형트럭)"],
        "display_name": "중형트럭 운영기준",
        "has_image": False,
    },
}

SUBTASK_TO_DOMAIN = {}
for domain, info in DOMAIN_SUBTASK_MAP.items():
    for st in info["subtasks"]:
        SUBTASK_TO_DOMAIN[st] = domain


##########################################################################
# Pipe 클래스
##########################################################################

class Pipe:
    class Valves(BaseModel):
        # ── 공통 ──
        CHROMA_PORT: int = 8800
        CHROMA_IP: str = "172.18.237.81"
        LLM_MODEL_NAME: str = "gpt-oss-120b"
        EMBED_PATH: str = "/data1/embedding/kure"

        # ── 운영기준 설정 ──
        BASE_IMG_URL: str = "https://ai.wooricap.com/static/auto_oper_images/"
        STANDARD_COLLECTION_NAME_IMAGE: str = "auto_oper_standard_image"
        STANDARD_COLLECTION_NAME_TEXT: str = "auto_oper_standard_text"
        EMBEDDING_K: int = 5
        REACT_MAX_ITERATIONS: int = 3
        ENABLE_FOLLOWUPS: bool = True
        ENABLE_HYBRID_SEARCH: bool = True
        RRF_K: int = 60
        RRF_VECTOR_WEIGHT: float = 0.7
        RRF_BM25_WEIGHT: float = 0.3
        ENABLE_GROUNDING_CHECK: bool = True

        # ── FAQ 설정 ──
        ENABLE_FAQ_SEARCH: bool = True
        ENABLE_FAQ_WIKI_SYNC: bool = True
        FAQ_COLLECTION_NAME: str = "faq_auto_oper"
        FAQ_EMBEDDING_K: int = 3
        FAQ_IMG_URL: str = "https://ai.wooricap.com/static/auto_oper_image/"
        FAQ_STATIC_IMG_PATH: str = (
            "/home/ubuntu/anaconda3/envs/webui/lib/python3.11/site-packages/open_webui/static/auto_oper_image/"
        )
        OUTLINE_API_KEY: str = "ol_api_GY0YveRgZ5bRYgOW0oV3ybgYCVJQh8gpMztEig"
        OUTLINE_URL: str = "https://util.wooricap.com:7443"
        FAQ_WIKI_COLLECTION_ID: str = "4efc49c2-f8c0-4f4e-b0f2-b5b926864acd"

    def __init__(self):
        self.valves = self.Valves()
        self._state_by_user: Dict[str, dict] = {}
        self._state_lock = asyncio.Lock()
        self._state_ttl_sec = 60 * 60

        # ── ChromaDB 클라이언트 ──
        chroma_client = chromadb.HttpClient(
            host=self.valves.CHROMA_IP, port=self.valves.CHROMA_PORT
        )

        # ── 운영기준: 임베딩 + 컬렉션 ──
        embedding_func = SentenceTransformerEmbeddings(
            model_name=self.valves.EMBED_PATH, model_kwargs={"device": "cpu"}
        )

        self.chroma_db_text = Chroma(
            client=chroma_client,
            collection_name=self.valves.STANDARD_COLLECTION_NAME_TEXT,
            embedding_function=embedding_func,
        )

        self.chroma_db_image = Chroma(
            client=chroma_client,
            collection_name=self.valves.STANDARD_COLLECTION_NAME_IMAGE,
            embedding_function=embedding_func,
        )

        # ── BM25 인덱스 구축 (하이브리드 검색용) ──
        self._bm25_indices = {}
        self._bm25_docs = {}

        if self.valves.ENABLE_HYBRID_SEARCH:
            try:
                raw_collection = chroma_client.get_collection(
                    self.valves.STANDARD_COLLECTION_NAME_TEXT
                )
                all_data = raw_collection.get(include=["documents", "metadatas"])

                subtask_groups: Dict[str, list] = {}
                for doc_text, metadata in zip(
                    all_data["documents"], all_data["metadatas"]
                ):
                    st = metadata.get("subtask", "")
                    if st not in subtask_groups:
                        subtask_groups[st] = []
                    subtask_groups[st].append(
                        Document(page_content=doc_text, metadata=metadata)
                    )

                for st, docs in subtask_groups.items():
                    tokenized_corpus = [d.page_content.split() for d in docs]
                    self._bm25_indices[st] = BM25Okapi(tokenized_corpus)
                    self._bm25_docs[st] = docs

                print(f"BM25 인덱스 구축 완료: {len(subtask_groups)}개 subtask, "
                      f"총 {sum(len(d) for d in subtask_groups.values())}건 문서")
            except Exception as e:
                print(f"BM25 인덱스 구축 실패 (벡터 검색만 사용): {e}")

        # ── FAQ: wiki sync + 컬렉션 초기화 ──
        self.chroma_db_faq = None

        if self.valves.ENABLE_FAQ_SEARCH:
            # ChromaDB 네이티브 임베딩 함수 (upsert용, kure 통일)
            self._faq_chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.valves.EMBED_PATH, device="cpu"
            )

            if self.valves.ENABLE_FAQ_WIKI_SYNC:
                try:
                    documents = self._proc_wiki_faq()
                    self._save_to_chroma(documents, self.valves.FAQ_COLLECTION_NAME)
                    print(f"FAQ wiki sync 완료: {len(documents)}건 문서")
                except Exception as e:
                    print(f"FAQ wiki sync 실패: {e}")

            try:
                self.chroma_db_faq = Chroma(
                    client=chroma_client,
                    collection_name=self.valves.FAQ_COLLECTION_NAME,
                    embedding_function=embedding_func,
                )
                print(f"FAQ 컬렉션 초기화 완료: {self.valves.FAQ_COLLECTION_NAME}")
            except Exception as e:
                print(f"FAQ 컬렉션 초기화 실패 (FAQ 검색 비활성화): {e}")
                self.chroma_db_faq = None

        # ── ChromaDB 클라이언트 보관 (save_to_chroma에서 사용) ──
        self._chroma_client = chroma_client

        print("통합 에이전트 Pipe 등록 완료! (v1 - FAQ 우선 + 운영기준 폴백)")

    # =================================================================
    # pipe() - 메인 엔트리포인트
    # ---------------------------------------------------------------
    # 흐름:
    #   1. _classify()       : 도메인/서브태스크 분류 (LLM 1회)
    #   2. _requery()        : 분류 불가 시 재질의 (LLM 1회)
    #   3. _faq_search()     : FAQ 벡터 검색
    #   4. _evaluate_faq()   : FAQ 충분성 판단 (LLM 1회)
    #   5a. FAQ 충분 → _generate(faq) : FAQ 답변 생성
    #   5b. FAQ 불충분 → _react_agent() → _generate(standard)
    # =================================================================
    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] | None = None,
        __user__: dict = None,
        __request__: Request = None,
    ):
        try:
            start_time = time.time()
            user_id = self._get_user_id(body, __user__)
            user_message = get_last_user_message(body["messages"])
            messages = body["messages"]
            send_status = get_send_status(__event_emitter__)

            user = Users.get_user_by_id(__user__["id"])

            user_msg_count = sum(1 for m in messages if m.get("role") == "user")
            is_new_chat = user_msg_count <= 1

            now = time.time()
            async with self._state_lock:
                for k, v in list(self._state_by_user.items()):
                    if now - v.get("ts", 0) > self._state_ttl_sec:
                        del self._state_by_user[k]

                if is_new_chat and user_id in self._state_by_user:
                    del self._state_by_user[user_id]

                user_state = self._state_by_user.get(user_id, {})
                last_classified = user_state.get("last_classified", "")
                last_domain = user_state.get("last_domain", "")

            state = {
                "user_message": user_message,
                "messages": messages,
                "user_id": user_id,
                "event_emitter": __event_emitter__,
                "send_status": send_status,
                "start_time": start_time,
                "__request__": __request__,
                "user": user,
                # _classify 결과
                "domain": "",
                "subtask": "",
                "is_requery": False,
                "requery_reason": "",
                "requery_detail": "",
                # _react_agent 결과
                "filtered_docs": [],
                # FAQ 결과
                "faq_docs": [],
                "faq_sufficient": False,
                "answer_source": "standard",
                # 멀티턴
                "is_first_entry": False,
                "last_classified": last_classified,
                "last_domain": last_domain,
            }

            # ── 1단계: FAQ 먼저 검색 + 충분성 판단 ──
            if self.valves.ENABLE_FAQ_SEARCH and self.chroma_db_faq:
                state = await self._faq_search(state)
                state = await self._evaluate_faq(state)

            if state["faq_sufficient"]:
                # FAQ로 충분 → classify 스킵, 바로 답변
                state["filtered_docs"] = state["faq_docs"]
                state["answer_source"] = "faq"
                return await self._generate(state)

            # ── 2단계: FAQ 불충분 → 분류 ──
            state = await self._classify(state)

            # ── 3단계: 재질의 분기 ──
            if state["is_requery"]:
                return await self._requery(state)

            # ── 4단계: ReAct 운영기준 검색 ──
            state["answer_source"] = "standard"
            state = await self._react_agent(state)

            # ── 5단계: 답변 생성 ──
            return await self._generate(state)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"exception :: {e}\n라인: {exc_tb.tb_lineno}"
            send_status = get_send_status(__event_emitter__)
            await send_status(status_message=f"오류 발생: {e}", done=True)
            return f"\n\n#### [Error]\n{error_message}\n"

    # =================================================================
    # _faq_search() - FAQ 컬렉션 벡터 검색
    # =================================================================
    async def _faq_search(self, state: dict) -> dict:
        send_status = state["send_status"]

        if not self.chroma_db_faq:
            state["faq_docs"] = []
            return state

        await send_status(status_message="FAQ 검색 중...", done=False)

        try:
            faq_docs = self.chroma_db_faq.similarity_search(
                query=state["user_message"],
                k=self.valves.FAQ_EMBEDDING_K,
            )
            state["faq_docs"] = faq_docs
        except Exception as e:
            print(f"FAQ 검색 실패: {e}")
            state["faq_docs"] = []

        return state

    # =================================================================
    # _evaluate_faq() - FAQ 충분성 LLM 판단
    # =================================================================
    async def _evaluate_faq(self, state: dict) -> dict:
        faq_docs = state.get("faq_docs", [])

        if not faq_docs:
            state["faq_sufficient"] = False
            return state

        send_status = state["send_status"]
        await send_status(status_message="FAQ 적합성 판단 중...", done=False)

        faq_context = "\n\n".join(
            [f"[FAQ {i+1}]\n{doc.page_content}" for i, doc in enumerate(faq_docs)]
        )

        prompt = FAQ_EVALUATE_PROMPT.format(
            faq_context=faq_context,
            question=state["user_message"],
        )

        try:
            result = await self._llm_call(prompt, state)
            cleaned = re.sub(r"```(?:json)?", "", result).strip().strip("`")
            data = json.loads(cleaned)
            state["faq_sufficient"] = bool(data.get("sufficient", False))
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"FAQ 판단 파싱 실패 (운영기준 폴백): {e}")
            state["faq_sufficient"] = False

        return state

    # =================================================================
    # _classify() - 도메인/서브태스크 분류
    # =================================================================
    async def _classify(self, state: dict) -> dict:
        send_status = state["send_status"]
        await send_status(status_message="질문 분류 중 ...", done=False)

        history_summary = self._build_history_summary(state["messages"])

        prompt = CLASSIFY_PROMPT.format(
            history=history_summary,
            last_domain=state["last_domain"] or "없음",
            last_subtask=state["last_classified"] or "없음",
            question=state["user_message"],
        )

        content = await self._llm_call(prompt, state)
        domain, subtask, requery_reason, requery_detail = self._parse_classification(content)

        is_first_entry = (
            state["last_classified"] == ""
            or state["last_classified"] != subtask
        )

        state["domain"] = domain
        state["subtask"] = subtask
        state["is_requery"] = domain == "재질의"
        state["is_first_entry"] = is_first_entry
        state["requery_reason"] = requery_reason
        state["requery_detail"] = requery_detail

        if not state["is_requery"]:
            display = DOMAIN_SUBTASK_MAP.get(domain, {}).get(
                "display_name", domain
            )
            await send_status(
                status_message=f"분류 결과 : {display} > {subtask}", done=True
            )
            await self._update_state(state["user_id"], domain, subtask)

        return state

    # =================================================================
    # _react_agent() - ReAct 루프
    # =================================================================
    async def _react_agent(self, state: dict) -> dict:
        observations = []
        seen_contents = set()
        send_status = state["send_status"]

        for i in range(self.valves.REACT_MAX_ITERATIONS):
            await send_status(
                status_message=f"ReAct 에이전트 실행 중... (반복 {i + 1}/{self.valves.REACT_MAX_ITERATIONS})",
                done=False,
            )

            decision = await self._react_step(state, observations)

            if decision["action"] == "FINISH":
                relevant_indices = decision.get("relevant_doc_indices", [])
                all_docs = []
                for obs in observations:
                    all_docs.extend(obs["docs"])

                if relevant_indices and all_docs:
                    state["filtered_docs"] = [
                        all_docs[idx] for idx in relevant_indices
                        if 0 <= idx < len(all_docs)
                    ]
                else:
                    state["filtered_docs"] = all_docs

                if not state["filtered_docs"] and all_docs:
                    state["filtered_docs"] = all_docs
                break

            elif decision["action"] == "SEARCH":
                query = decision["query"]
                subtask = decision["subtask"]

                await send_status(
                    status_message=f"{subtask} 검색 중: \"{query[:30]}...\"" if len(query) > 30 else f"{subtask} 검색 중: \"{query}\"",
                    done=False,
                )

                raw_docs = self._execute_tool(query, subtask)

                docs = []
                for d in raw_docs:
                    if d.page_content not in seen_contents:
                        seen_contents.add(d.page_content)
                        docs.append(d)

                offset = sum(len(obs["docs"]) for obs in observations)

                observations.append({
                    "query": query,
                    "subtask": subtask,
                    "docs": docs,
                    "doc_summaries": [
                        f"[문서{offset + j}] {d.page_content[:200]}"
                        for j, d in enumerate(docs)
                    ],
                })
        else:
            all_docs = []
            for obs in observations:
                all_docs.extend(obs["docs"])
            state["filtered_docs"] = all_docs

        await send_status(
            status_message=f"관련 문서 {len(state['filtered_docs'])}건 수집 완료",
            done=True,
        )
        return state

    # =================================================================
    # _react_step() - ReAct 1스텝 판단
    # =================================================================
    async def _react_step(self, state: dict, observations: list) -> dict:
        domain = state["domain"]
        subtask = state["subtask"]
        domain_info = DOMAIN_SUBTASK_MAP.get(domain, {})
        available_subtasks = ", ".join(domain_info.get("subtasks", [subtask]))

        if observations:
            obs_text_parts = []
            for idx, obs in enumerate(observations):
                obs_text_parts.append(
                    f"--- 검색 {idx + 1}: query=\"{obs['query']}\", subtask={obs['subtask']} ---"
                )
                obs_text_parts.extend(obs["doc_summaries"])
            obs_text = "\n".join(obs_text_parts)
        else:
            obs_text = "없음 (첫 검색)"

        prompt = REACT_STEP_PROMPT.format(
            domain=domain_info.get("display_name", domain),
            subtask=subtask,
            available_subtasks=available_subtasks,
            observations=obs_text,
            question=state["user_message"],
        )

        content = await self._llm_call(prompt, state)
        return self._parse_react_decision(content, state)

    # =================================================================
    # _execute_tool() - 하이브리드 검색 (벡터 + BM25 + RRF)
    # =================================================================
    def _execute_tool(self, query: str, subtask: str) -> list:
        k = self.valves.EMBEDDING_K

        if not self.valves.ENABLE_HYBRID_SEARCH or subtask not in self._bm25_indices:
            return self.chroma_db_text.similarity_search(
                query=query, k=k, filter={"subtask": subtask},
            )

        search_k = k * 2

        vector_results = self.chroma_db_text.similarity_search(
            query=query, k=search_k, filter={"subtask": subtask},
        )

        tokenized_query = query.split()
        bm25 = self._bm25_indices[subtask]
        bm25_docs_all = self._bm25_docs[subtask]
        scores = bm25.get_scores(tokenized_query)

        scored_indices = [
            (idx, sc) for idx, sc in enumerate(scores) if sc > 0
        ]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        bm25_results = [bm25_docs_all[idx] for idx, _ in scored_indices[:search_k]]

        return self._rrf_merge(vector_results, bm25_results, k)

    # =================================================================
    # _rrf_merge() - Reciprocal Rank Fusion
    # =================================================================
    def _rrf_merge(self, vector_docs: list, bm25_docs: list, k: int) -> list:
        rrf_k = self.valves.RRF_K
        w_vec = self.valves.RRF_VECTOR_WEIGHT
        w_bm25 = self.valves.RRF_BM25_WEIGHT
        doc_scores = {}
        doc_objects = {}

        for rank, doc in enumerate(vector_docs):
            key = doc.page_content
            doc_scores[key] = doc_scores.get(key, 0) + w_vec / (rrf_k + rank + 1)
            doc_objects[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            doc_scores[key] = doc_scores.get(key, 0) + w_bm25 / (rrf_k + rank + 1)
            if key not in doc_objects:
                doc_objects[key] = doc

        sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        return [doc_objects[key] for key in sorted_keys[:k]]

    # =================================================================
    # _format_citation_html() - 출처 문서를 HTML 카드로 포맷팅
    # =================================================================
    def _format_citation_html(self, doc, idx: int, answer_source: str) -> str:
        if answer_source == "faq":
            title = doc.metadata.get("질문", doc.metadata.get("title", f"FAQ {idx}"))
            source_url = doc.metadata.get("source", "")
            content = doc.page_content  # 이미 HTML (markdown() 변환 완료)

            source_link = ""
            if source_url:
                source_link = (
                    '<div style="margin-top:12px;padding-top:8px;border-top:1px solid #e5e7eb;font-size:12px">'
                    f'<a href="{source_url}" target="_blank" style="color:#4f46e5;text-decoration:none">원문 보기</a>'
                    '</div>'
                )

            return (
                '<div style="font-family:-apple-system,sans-serif;font-size:14px;line-height:1.6">'
                '<div style="background:#eef2ff;padding:10px 14px;border-radius:8px;margin-bottom:10px;border-left:4px solid #4f46e5">'
                f'<strong style="color:#4f46e5">Q.</strong> {title}'
                '</div>'
                f'<div style="padding:0 4px">{content}</div>'
                f'{source_link}'
                '</div>'
            )
        else:
            content_html = markdown(doc.page_content)
            return (
                '<div style="font-family:-apple-system,sans-serif;font-size:14px;line-height:1.6">'
                f'<div style="padding:0 4px">{content_html}</div>'
                '</div>'
            )

    # =================================================================
    # _generate() - 최종 답변 생성 (FAQ/운영기준 통합)
    # ---------------------------------------------------------------
    # answer_source에 따라 프롬프트 분기:
    #   - "faq": FAQ_GENERATE_PROMPT (이미지 태그 처리 포함)
    #   - "standard": GENERATE_PROMPT (금리표 해석 등 운영기준 전용)
    # =================================================================
    async def _generate(self, state: dict):
        send_status = state["send_status"]
        domain = state["domain"]
        subtask = state["subtask"]
        filtered_docs = state["filtered_docs"]
        user_message = state["user_message"]
        messages = state["messages"]
        answer_source = state.get("answer_source", "standard")
        display_name = DOMAIN_SUBTASK_MAP.get(domain, {}).get(
            "display_name", domain
        )

        # ── 운영기준 전용: 이미지 경로 분기 ──
        if answer_source == "standard":
            if state["is_first_entry"] and DOMAIN_SUBTASK_MAP.get(domain, {}).get("has_image", False):
                return await self._generate_with_image(state)

        # ── 참고자료 컨텍스트 구성 ──
        context = "\n\n".join(
            [f"[참고자료 {i+1}]\n{doc.page_content}" for i, doc in enumerate(filtered_docs)]
        )

        # ── 프롬프트 분기 ──
        if answer_source == "faq":
            prompt = FAQ_GENERATE_PROMPT.format(
                domain=display_name,
                subtask=subtask,
                history=str(messages[-6:]) if len(messages) > 6 else str(messages),
                context=context,
                question=user_message,
            )
        else:
            prompt = GENERATE_PROMPT.format(
                domain=display_name,
                subtask=subtask,
                history=str(messages[-6:]) if len(messages) > 6 else str(messages),
                context=context,
                question=user_message,
            )

        # ── 출처(citation) 전송 (HTML 카드형) ──
        send_citation = get_send_citation(state["event_emitter"])
        for idx, doc in enumerate(filtered_docs, start=1):
            citation_html = self._format_citation_html(doc, idx, answer_source)
            title_text = doc.metadata.get("title", f"출처{idx}") if answer_source == "faq" else f"출처{idx}"
            await send_citation(
                url=f"출처{idx}",
                title=title_text,
                content=citation_html,
            )

        # ── 답변 생성 (non-streaming) ──
        await send_status(status_message="답변 생성 중...", done=False)
        llm_text = await self._llm_call(prompt, state)

        # ── 후속질문 + 신뢰도 병렬 실행 ──
        followup_task = None
        grounding_task = None
        if self.valves.ENABLE_FOLLOWUPS:
            followup_task = asyncio.create_task(
                self._generate_followups(state, llm_text)
            )
        if self.valves.ENABLE_GROUNDING_CHECK and llm_text:
            grounding_task = asyncio.create_task(
                self._check_grounding(state, llm_text)
            )

        if followup_task or grounding_task:
            await send_status(
                status_message="추천 질문 / 답변 신뢰도 확인 중...",
                done=False,
            )

        follow_ups = None
        grounding = None
        if followup_task:
            follow_ups = await followup_task
        if grounding_task:
            grounding = await grounding_task

        # ── 완료 시간 ──
        exe_time = time.time() - state["start_time"]
        await send_status(status_message=f"완료: {exe_time:.2f}초", done=True)

        # ── 최종 결과 조합 ──
        if answer_source == "faq":
            combined = f"#### [FAQ]\n{llm_text}"
        else:
            combined = f"#### [{display_name} > {subtask}]\n{llm_text}"

        if follow_ups:
            combined += "\n\n---\n**추천 질문:**\n"
            for idx, q in enumerate(follow_ups, 1):
                combined += f"{idx}. {q}\n"

        if grounding:
            score = grounding["score"]
            filled = int(score / 10)
            bar = "\u25a0" * filled + "\u25a1" * (10 - filled)
            combined += f"\n\n---\n**[답변 신뢰도: {bar} {score}%]** {grounding['reason']}"

        def stream_output():
            for char in combined:
                yield char

        return stream_output()

    # =================================================================
    # _generate_with_image() - 이미지 포함 답변 (운영기준 전용)
    # =================================================================
    async def _generate_with_image(self, state: dict):
        domain = state["domain"]
        subtask = state["subtask"]
        filtered_docs = state["filtered_docs"]
        user_message = state["user_message"]
        messages = state["messages"]
        display_name = DOMAIN_SUBTASK_MAP.get(domain, {}).get(
            "display_name", domain
        )

        image_results = self.chroma_db_image.similarity_search(
            query=subtask, k=1
        )

        image_url = ""
        if image_results:
            image_url = image_results[0].metadata.get("html_images", "")
        if not image_url:
            image_url = "이미지 검색 결과 없음"

        context = "\n\n".join(
            [f"[참고자료 {i+1}]\n{doc.page_content}" for i, doc in enumerate(filtered_docs)]
        )

        prompt = GENERATE_PROMPT.format(
            domain=display_name,
            subtask=subtask,
            history=str(messages[-6:]) if len(messages) > 6 else str(messages),
            context=context,
            question=user_message,
        )

        send_status = state["send_status"]

        send_citation = get_send_citation(state["event_emitter"])
        for idx, doc in enumerate(filtered_docs, start=1):
            citation_html = self._format_citation_html(doc, idx, "standard")
            await send_citation(
                url=f"출처{idx}",
                title=f"출처{idx}",
                content=citation_html,
            )

        await send_status(status_message="답변 생성 중...", done=False)
        llm_text = await self._llm_call(prompt, state)

        followup_task = None
        grounding_task = None
        if self.valves.ENABLE_FOLLOWUPS:
            followup_task = asyncio.create_task(
                self._generate_followups(state, llm_text)
            )
        if self.valves.ENABLE_GROUNDING_CHECK and llm_text:
            grounding_task = asyncio.create_task(
                self._check_grounding(state, llm_text)
            )

        if followup_task or grounding_task:
            await send_status(
                status_message="추천 질문 / 답변 신뢰도 확인 중...",
                done=False,
            )

        follow_ups = None
        grounding = None
        if followup_task:
            follow_ups = await followup_task
        if grounding_task:
            grounding = await grounding_task

        exe_time = time.time() - state["start_time"]
        await send_status(status_message=f"완료: {exe_time:.2f}초", done=True)

        combined = f"#### [{display_name} > {subtask}]\n{image_url}\n{llm_text}"

        if follow_ups:
            combined += "\n\n---\n**추천 질문:**\n"
            for idx, q in enumerate(follow_ups, 1):
                combined += f"{idx}. {q}\n"

        if grounding:
            score = grounding["score"]
            filled = int(score / 10)
            bar = "\u25a0" * filled + "\u25a1" * (10 - filled)
            combined += f"\n\n---\n**[답변 신뢰도: {bar} {score}%]** {grounding['reason']}"

        def stream_output():
            for char in combined:
                yield char

        return stream_output()

    # =================================================================
    # _generate_followups() - 후속 추천 질문 생성
    # =================================================================
    async def _generate_followups(self, state: dict, answer: str) -> list:
        try:
            domain = state["domain"]
            subtask = state["subtask"]
            filtered_docs = state.get("filtered_docs", [])
            domain_info = DOMAIN_SUBTASK_MAP.get(domain, {})
            display_name = domain_info.get("display_name", domain)

            context = "\n\n".join(
                [f"[참고자료 {i+1}]\n{doc.page_content}" for i, doc in enumerate(filtered_docs)]
            )

            all_subtasks = domain_info.get("subtasks", [])
            other_subtasks = [s for s in all_subtasks if s != subtask]
            other_subtasks_str = ", ".join(other_subtasks) if other_subtasks else "없음"

            prompt = FOLLOWUP_PROMPT.format(
                domain=display_name,
                subtask=subtask,
                other_subtasks=other_subtasks_str,
                context=context,
                question=state["user_message"],
                answer=answer[:2000],
            )

            content = await self._llm_call(prompt, state)

            cleaned = re.sub(r"```(?:json)?", "", content).strip().strip("`")
            data = json.loads(cleaned)
            follow_ups = data.get("follow_ups", [])

            if isinstance(follow_ups, list) and len(follow_ups) >= 1:
                return [str(q) for q in follow_ups[:3]]
            return []

        except Exception as e:
            print(f"[Followup generation error] {e}")
            return []

    # =================================================================
    # _check_grounding() - 답변 신뢰도 검증
    # =================================================================
    async def _check_grounding(self, state: dict, answer: str) -> dict | None:
        try:
            filtered_docs = state.get("filtered_docs", [])
            context = "\n\n".join(
                [f"[참고자료 {i+1}]\n{doc.page_content}" for i, doc in enumerate(filtered_docs)]
            )

            prompt = GROUNDING_CHECK_PROMPT.format(
                context=context,
                question=state["user_message"],
                answer=answer[:3000],
            )

            content = await self._llm_call(prompt, state)

            cleaned = re.sub(r"```(?:json)?", "", content).strip().strip("`")
            data = json.loads(cleaned)
            score = float(data.get("score", 0))
            reason = str(data.get("reason", ""))

            if 0.0 <= score <= 100.0:
                return {"score": round(score, 1), "reason": reason}
            return None

        except Exception as e:
            print(f"[Grounding check error] {e}")
            return None

    # =================================================================
    # _requery() - 재질의 응답 생성
    # =================================================================
    async def _requery(self, state: dict):
        send_status = state["send_status"]
        await send_status(status_message="재질의 응답 생성 중...", done=False)

        prompt = REQUERY_PROMPT.format(
            history=str(state["messages"][-6:]),
            question=state["user_message"],
            requery_reason=state.get("requery_reason", "모호"),
            requery_detail=state.get("requery_detail", ""),
        )

        end_time = time.time()
        exe_time = end_time - state["start_time"]
        await send_status(status_message=f"완료: {exe_time:.2f}초", done=True)

        data_json = {
            "model": self.valves.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0,
        }
        response = await generate_chat_completion(
            state["__request__"], data_json, state["user"]
        )
        return response

    # =================================================================
    # 유틸리티 함수
    # =================================================================

    async def _llm_call(self, prompt: str, state: dict) -> str:
        data_json = {
            "model": self.valves.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0,
        }
        response = await generate_chat_completion(
            state["__request__"], data_json, state["user"]
        )
        return response["choices"][0]["message"]["content"]

    def _get_user_id(self, body: dict, __user__: dict | None = None) -> str:
        if isinstance(__user__, dict) and __user__.get("id"):
            return __user__["id"]
        return (body.get("user") or {}).get("id") or "anonymous"

    async def _update_state(self, user_id: str, domain: str, subtask: str) -> None:
        async with self._state_lock:
            self._state_by_user[user_id] = {
                "last_classified": subtask,
                "last_domain": domain,
                "ts": time.time(),
            }

    def _build_history_summary(self, messages: list) -> str:
        if len(messages) <= 1:
            return "없음 (첫 질문)"

        history_parts = []
        recent = messages[-7:-1]
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if len(content) > 300:
                content = content[:150] + "\n...(중략)...\n" + content[-150:]
            history_parts.append(f"[{role}] {content}")

        return "\n".join(history_parts) if history_parts else "없음 (첫 질문)"

    def _parse_classification(self, content: str) -> tuple:
        try:
            cleaned = re.sub(r"```(?:json)?", "", content).strip().strip("`")
            data = json.loads(cleaned)
            domain = data.get("domain", "재질의")
            subtask = data.get("subtask", "재질의")
            requery_reason = data.get("requery_reason", "모호")
            requery_detail = data.get("requery_detail", "")

            if domain in DOMAIN_SUBTASK_MAP:
                valid_subtasks = DOMAIN_SUBTASK_MAP[domain]["subtasks"]
                if subtask not in valid_subtasks:
                    for vs in valid_subtasks:
                        if subtask.replace("(", "").replace(")", "") in vs:
                            subtask = vs
                            break
                    else:
                        if len(valid_subtasks) == 1:
                            subtask = valid_subtasks[0]
                return domain, subtask, "", ""
            else:
                return "재질의", "재질의", requery_reason, requery_detail
        except (json.JSONDecodeError, KeyError, AttributeError):
            for domain, info in DOMAIN_SUBTASK_MAP.items():
                for st in info["subtasks"]:
                    if st in content:
                        return domain, st, "", ""
            return "재질의", "재질의", "모호", ""

    def _parse_react_decision(self, content: str, state: dict) -> dict:
        try:
            cleaned = re.sub(r"```(?:json)?", "", content).strip().strip("`")
            data = json.loads(cleaned)
            action = data.get("action", "FINISH")

            if action == "SEARCH":
                query = data.get("query", state["user_message"])
                subtask = data.get("subtask", state["subtask"])

                domain_info = DOMAIN_SUBTASK_MAP.get(state["domain"], {})
                valid_subtasks = domain_info.get("subtasks", [])
                if subtask not in valid_subtasks:
                    for vs in valid_subtasks:
                        if subtask.replace("(", "").replace(")", "") in vs:
                            subtask = vs
                            break
                    else:
                        subtask = state["subtask"]

                return {"action": "SEARCH", "query": query, "subtask": subtask}

            elif action == "FINISH":
                relevant_indices = data.get("relevant_doc_indices", [])
                return {
                    "action": "FINISH",
                    "relevant_doc_indices": [
                        i for i in relevant_indices if isinstance(i, int)
                    ],
                }

            else:
                return {"action": "FINISH", "relevant_doc_indices": []}

        except (json.JSONDecodeError, KeyError, AttributeError):
            return {
                "action": "SEARCH",
                "query": state["user_message"],
                "subtask": state["subtask"],
            }

    # =================================================================
    # FAQ 데이터 처리 메서드 (FAQ v11에서 이식)
    # =================================================================

    def _proc_wiki_faq(self) -> List[Document]:
        """Outline wiki에서 FAQ 문서 다운로드 + 이미지 처리 → Document 리스트"""
        os.makedirs(self.valves.FAQ_STATIC_IMG_PATH, exist_ok=True)
        self._delete_all_files_in_directory(self.valves.FAQ_STATIC_IMG_PATH)

        response_data = self._list_docs(self.valves.FAQ_WIKI_COLLECTION_ID)

        documents = []
        for data in response_data:
            doc_id = data["id"]
            title = data["title"]
            text = data["text"]
            source = self.valves.OUTLINE_URL + data["url"]

            urls = self._extract_image_urls(text)
            html_text = markdown("**질문** : \n" + title + "\n\n" + text)

            modified_text = text
            for idx, url in enumerate(urls):
                image_id = self._extract_id(url)
                file_name = self._download_image(url, doc_id, image_id)

                modified_text = self._replace_image_url_to_tag(modified_text, file_name)
                modified_text = self._process_content_with_images(modified_text)
                html_text = markdown("**질문** : \n" + title + "\n\n" + modified_text)

            document = Document(
                page_content=html_text.strip().replace("\n", ""),
                metadata={
                    "title": title.strip().replace("\n", ""),
                    "질문": title.strip().replace("\n", ""),
                    "답변": modified_text,
                    "source": source,
                },
            )

            documents.append(document)

        return documents

    def _save_to_chroma(self, documents: List[Document], collection_name: str):
        """Document 리스트를 ChromaDB 컬렉션에 저장 (기존 삭제 후 재생성)"""
        try:
            try:
                self._chroma_client.delete_collection(collection_name)
            except Exception:
                pass

            collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._faq_chroma_ef,
            )

            collection.upsert(
                documents=[doc.page_content for doc in documents],
                ids=[f"{collection_name}{i}" for i in range(len(documents))],
                metadatas=[doc.metadata for doc in documents],
            )
        except Exception as e:
            print(f"ChromaDB 저장 실패: {e}")

    def _safe_post(self, url: str, json_data: dict, max_retry: int = 8):
        """Outline API 호출 (429 Rate Limit 시 재시도)"""
        for attempt in range(max_retry):
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.valves.OUTLINE_API_KEY}",
            }

            r = requests.post(url, json=json_data, headers=headers, timeout=30)
            if r.status_code != 429:
                r.raise_for_status()
                return r
            wait = int(r.headers.get("Retry-After", "10"))
            jitter = random.uniform(0, 3)
            sleep_sec = wait + jitter if wait else 10 + attempt * 2
            print(f"[RateLimit] {url} → 재시도까지 {sleep_sec:.1f}s 대기")
            time.sleep(sleep_sec)
        raise RuntimeError(f"Too many retries for {url}")

    def _list_docs(self, collection_id: str, limit: int = 100) -> List[dict]:
        """Outline 컬렉션의 전체 문서 조회"""
        data = {"limit": limit}
        if collection_id:
            data["collectionId"] = collection_id
        docs, off = [], 0
        while True:
            r = self._safe_post(
                f"{self.valves.OUTLINE_URL}/api/documents.list", data | {"offset": off}
            )
            batch = r.json()["data"]
            if not batch:
                break
            docs.extend(batch)
            off += len(batch)
        return docs

    def _extract_image_urls(self, markdown_text: str) -> list:
        """Markdown 텍스트에서 이미지 URL 추출"""
        pattern = r"!\[\]\((.*?)\)"
        matches = re.findall(pattern, markdown_text)
        urls = []
        for match in matches:
            url = match.split()[0]
            urls.append(self.valves.OUTLINE_URL + url)
        return urls

    def _extract_id(self, url: str) -> str:
        """URL에서 이미지 ID 추출"""
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        return query_params.get("id", [None])[0]

    def _download_image(self, url: str, document_id: str, image_id: str) -> str:
        """이미지 다운로드 + 로컬 저장"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.OUTLINE_API_KEY}",
        }
        filename = f"{document_id}_{image_id}.png"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                filepath = os.path.join(self.valves.FAQ_STATIC_IMG_PATH, filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
        return filename

    def _replace_image_url_to_tag(self, text: str, file_name: str) -> str:
        """이미지 링크를 <ImageInWiki> 태그로 변환"""
        image_link_pattern = r"!\[\]\((.*?)\)"
        match = re.search(image_link_pattern, text)
        if match:
            replacement = f"<ImageInWiki>{file_name}</ImageInWiki>"
            return text.replace(match.group(0), replacement)
        return text

    def _process_content_with_images(self, content: str) -> str:
        """<ImageInWiki> 태그를 img HTML 태그로 변환"""
        pattern = re.compile(r"<ImageInWiki>(.*?)</ImageInWiki>")

        def replace_image_tag(match):
            image_name = match.group(1).strip()
            image_path = os.path.join(self.valves.FAQ_STATIC_IMG_PATH, image_name)
            return self._convert_image_to_url(image_path, image_name)

        return pattern.sub(replace_image_tag, content)

    def _convert_image_to_url(self, image_path: str, image_name: str) -> str:
        """파일 경로 → img HTML 태그"""
        if os.path.exists(image_path):
            return f'<div class="image-container"><img src="{self.valves.FAQ_IMG_URL}{image_name}" alt="{image_name}" class="fill-image"></div>'
        return "(이미지 파일을 찾을 수 없음)"

    def _delete_all_files_in_directory(self, directory_path: str):
        """디렉토리 내 파일 전체 삭제"""
        try:
            if not os.path.exists(directory_path):
                return
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"파일 삭제 실패: {e}")


###################################################################################
# 프롬프트 템플릿
###################################################################################

# ==========================================================================
# FAQ_EVALUATE_PROMPT - FAQ 충분성 판단
# ==========================================================================
FAQ_EVALUATE_PROMPT = """사용자 질문에 대해 아래 FAQ 자료만으로 정확하고 충분한 답변이 가능한지 판단하세요.

### FAQ 자료
{faq_context}

### 사용자 질문
{question}

### 판단 기준
1. FAQ에 질문과 **직접 관련된** 내용이 있는가?
2. FAQ 내용만으로 **완전한 답변**이 가능한가? (추측이나 보충 불필요)
3. **부분적으로만** 관련되거나 핵심 정보가 빠져있으면 "불충분"으로 판단

### 출력 형식 (JSON만 출력, 다른 텍스트 없이)
{{"sufficient": true또는false, "reason": "판단 근거를 한 문장으로 설명"}}"""


# ==========================================================================
# FAQ_GENERATE_PROMPT - FAQ 기반 답변 생성
# ==========================================================================
FAQ_GENERATE_PROMPT = """너는 JB우리캐피탈 오토운영팀 챗봇이야.
현재 업무 영역: {domain} > {subtask}

사용자에게 주어진 질문에 대해서 참고자료(FAQ)를 활용해서 적절히 답변해.

<지침>
1. 제공되는 참고자료를 참고하여 적합한 답변을 생성하고 DocumentID 같은 IT 단어는 사용하지마
2. 사용자 질문이 너무 광범위할 경우, 구체적으로 알려달라고 요청해.

<이미지 처리>
1. 참고자료에 이미지 태그(img src)가 포함된 경우, 해당 이미지 URL을 마크다운 이미지 형식으로 변환하여 답변에 포함해.
   양식: ![이미지](이미지URL)
2. 이미지가 없는 경우 URL을 생성하지 마.

<답변 근거 원칙 - 반드시 준수>
1. **참고자료에 있는 정보만** 사용하여 답변을 생성해. 참고자료에 없는 내용은 절대 포함하지 마.
2. 숫자, 비율, 등급, 금리, 기간, 한도 등 **구체적 수치는 반드시 참고자료에서 확인된 것만** 사용해.
3. 일반 상식이나 추측으로 답변을 보충하지 마. 참고자료에 없으면 "해당 정보는 현재 참고자료에서 확인되지 않습니다"라고 명시해.
4. 참고자료의 내용을 **왜곡하거나 확대 해석**하지 마. 원문의 의미를 그대로 전달해.

<답변 가독성>
1. 답변 작성시 가독성을 위해 문장이 끝나면 줄바꿈 처리하고, 답변이 길어지면 글머리 기호를 줘서 가독성을 높여줘.
2. 구분자를 줄때 마지막에 빈 구분자가 생성되는데, 빈 구분자가 발생하지 않도록해.
3. 정보 전달할때 중요한 내용에 대해서는 볼드 처리 해줘.

<추가 사항>
1. 인사 하지마. 바로 질문에 답변하도록해.
2. 참고자료에서 참고할 수 없는 경우에는 "참고자료를 찾을 수 없습니다. 오토운영팀에 문의해주세요"라고 답변해.
3. 너는 반드시 "한국어"로 답변해.

대화내용 : {history}

참고자료 : {context}

사용자 질문: {question}"""


# ==========================================================================
# CLASSIFY_PROMPT - 질문 분류
# ==========================================================================
CLASSIFY_PROMPT = """당신은 사용자의 질문을 **도메인**과 **서브태스크**로 분류하는 모델입니다.
아래 **도메인·서브태스크 정의**와 **분류 규칙**을 정확히 따르고, 가능한 경우 이전 대화 히스토리에서 최근 업무를 확인해 연속된 문의인지 판단하십시오.
분류하기 어려운 복잡한 질문이 들어올 수 있으므로, 차분히 고려해서 답변해.

### 도메인 및 서브태스크 정의

1. **중고승용** (subtask 5개)
   - (론/할부)
     - 정의: 일반고객에 대해 중고차 대출을 전액대출 형태(론)와 할부납부하는 형태(할부)로 상품이 구분됨
     - 키워드: 중고승용차, 구입대출, 코드, 영업채널, 제휴점, Direct, 기간, 개인, 법인, 금리등급, G/L금리, NEGO조정금리, NICE등급, 내국인, 외국인, 거점장, 증빙서류, HJSeg, 판촉수수료, 연체이자율, 슬라이딩
   - (임직원대출)
     - 정의: 일반고객 대출 외 임직원에 대한 대출이며, 일반 임직원과 딜러 상품이 별도 존재함
     - 키워드: 임직원대출, ESM, 딜러, NICE등급, 금리등급, 조정금리, 슬라이딩, 실적급수수료, 대출가능횟수
   - (신용구제)
     - 정의: 신용회복한 고객(신용구제 대상)을 대상으로 중고차 대출해주는 상품
     - 키워드: 신용구제, 개인회생, 신용회복, 시범운영, 월취급액, 연령, 상환방식
   - (Dual Offer)
     - 정의: 대출한도 높게 원하는 사람들한테 한도 늘려주고 금리 높게 해주는 상품
     - 키워드: Dual, 세부코드, 최대개월수, 최저금리, 중도상환수료율
   - (엔카)
     - 정의: 엔카 다이렉트(플랫폼) 상담조회 이력 보유 고객으로 상담 분개건을 처리하는 상품
     - 키워드: 엔카, 엔카다이렉트, 무수수료, 플랫폼, 상담조회, 분개, 금리등급, 네고, 가이드라인금리

2. **전략금융** (subtask 5개)
   - (재고금융)
     - 정의: 중고차 매매상사를 대상으로 차량 재고를 담보로 대출·한도·이자·LTV 등을 제공하는 금융상품
     - 키워드: 재고금융, 약정서, 실사, 한도산정, 등급관리, 연체, 사후관리
   - (제휴점 운영자금)
     - 정의: 제휴점(협력 점포)의 월 매출 규모에 따라 평가된 한도 내에서 제공되는 운용 대출 상품
     - 키워드: 제휴점, 운영자금, 매출, 목표달성, 패널티
   - (매매상사 운영자금)
     - 정의: 최근 3개월 최고 판매월의 판매대수와 NICE CB 점수에 따라 한도가 결정되는 대출상품
     - 키워드: 매매상사, 판매대수, NICE CB, 영업본부장 전결
   - (운영자금 자금용도 기준)
     - 정의: 대출금 사용 목적별로 요구되는 증빙서류·증빙방법·미이행 시 적용되는 Penalty를 명시한 규정
     - 키워드: 자금용도, 타사대환, 증빙서류, 기한이익상실
   - (임차보증금)
     - 정의: 임차보증금 관련 금융상품
     - 키워드: 임차보증금, 금리

3. **중고리스** (subtask 1개)
   - (중고리스)
     - 정의: 중고차량에 대한 운용리스/금융리스 상품
     - 키워드: 운용리스, 금융리스, 잔가, 잔가율, 그룹 I-P, 수입차, 보험, 세금

4. **중형트럭** (subtask 1개)
   - (중형트럭)
     - 정의: 중형트럭에 대한 할부/대출 상품
     - 키워드: 중형트럭, 할부, 대출, LTV, 특장, 주행거리, 예외협의

### 분류 규칙
1. 사용자 질문과 키워드/정의를 대조하여 가장 적합한 도메인 + 서브태스크를 선택
2. **멀티턴 유지 (최우선 규칙)**: 이전 서브태스크가 존재하면, **다른 서브태스크 이름을 명시적으로 언급**하지 않는 한 이전 서브태스크를 유지한다.
   - "네고", "금리", "한도", "조건", "수수료", "등급" 등 **여러 서브태스크에 공통으로 사용되는 일반 용어**만으로는 카테고리를 전환하지 않는다.
   - 카테고리 전환은 **서브태스크 고유 이름**(론/할부, 엔카, 신용구제, 재고금융 등)이 질문에 직접 언급될 때만 수행한다.
   - 유지 예시:
     - 1) "엔카 운영기준 알려줘" → (엔카)  2) "네고 기준은?" → (엔카) ✅ 유지
     - 1) "론할부 운영기준 알려줘" → (론/할부)  2) "nice 등급은?" → (론/할부) ✅ 유지
     - 1) "론할부 운영기준 알려줘" → (론/할부)  2) "금리 알려줘" → (론/할부) ✅ 유지
   - 전환 예시:
     - 1) "론할부 운영기준 알려줘" → (론/할부)  2) "신용구제 금리는?" → (신용구제) ✅ 전환 (신용구제 명시)
     - 1) "엔카 운영기준 알려줘" → (엔카)  2) "재고금융 한도는?" → (재고금융) ✅ 전환 (재고금융 명시)
3. **재질의 판단**: 아래 3가지 경우에 해당하면 domain="재질의", subtask="재질의"로 분류하고, requery_reason을 반드시 포함하세요.
   - **"업무외"**: 지원 업무 영역(중고승용/전략금융/중고리스/중형트럭)과 전혀 관련 없는 질문 (예: "날씨 어때?", "점심 추천해줘")
   - **"모호"**: 업무와 관련될 수 있으나 범위가 너무 넓거나 도메인/서브태스크를 특정할 수 없는 질문 (예: "운영기준 알려줘", "금리 알려줘")
   - **"정보부족"**: 도메인은 유추 가능하나, 정확한 답변을 위해 추가 정보(상품유형, 고객유형, 등급, 금액 등)가 필요한 질문 (예: "대출 한도가 얼마야?", "수수료율 알려줘")

### 이전 대화 맥락
- 이전 도메인: {last_domain}
- 이전 서브태스크: {last_subtask}

### 이전 대화 히스토리
{history}

### 현재 질문
{question}

### 출력 형식 (JSON만 출력, 다른 텍스트 없이)
분류 성공 시:
```json
{{"domain": "중고승용|전략금융|중고리스|중형트럭", "subtask": "(론/할부)|(임직원대출)|..."}}
```

재질의 시 (반드시 requery_reason 포함):
```json
{{"domain": "재질의", "subtask": "재질의", "requery_reason": "업무외|모호|정보부족", "requery_detail": "재질의 사유를 한 문장으로 설명"}}
```"""


# ==========================================================================
# REACT_STEP_PROMPT - ReAct 1스텝 판단
# ==========================================================================
REACT_STEP_PROMPT = """당신은 JB우리캐피탈 오토운영팀의 정보 검색 에이전트입니다.
사용자 질문에 답하기 위해 벡터 검색 도구를 사용하여 관련 문서를 수집합니다.

### 현재 도메인: {domain}
### 분류된 서브태스크: {subtask}
### 사용 가능한 서브태스크: {available_subtasks}

### 도구
- vector_search(query, subtask): 지식베이스에서 관련 문서를 검색합니다.
  - query: 검색어 (한국어, 구체적이고 핵심 키워드 중심으로)
  - subtask: 위 서브태스크 중 하나

### 이전 검색 결과
{observations}

### 사용자 질문
{question}

### 지침
1. 이전 검색 결과가 없으면, 사용자 질문에 가장 적합한 검색어와 서브태스크로 검색하세요.
   - 기본적으로 분류된 서브태스크를 사용하되, 질문 내용에 따라 다른 서브태스크가 적합하면 변경 가능합니다.
2. 이전 검색 결과가 있으면, 결과를 검토하고:
   - 질문에 답하기에 **충분한 정보**가 있으면 → FINISH를 선택하고 관련 문서 번호를 지정하세요.
   - 정보가 **부족**하거나 **다른 관점**의 검색이 필요하면 → 다른 검색어나 서브태스크로 추가 검색하세요.
3. 검색어는 사용자 질문의 핵심 키워드를 추출하여 구체적으로 작성하세요.
4. 동일한 검색어와 서브태스크로 반복 검색하지 마세요.

### 출력 형식 (JSON만 출력, 다른 텍스트 없이)
검색이 필요한 경우:
{{"action": "SEARCH", "thought": "검색 이유를 간단히 설명", "query": "검색어", "subtask": "(서브태스크명)"}}

충분한 정보가 수집된 경우:
{{"action": "FINISH", "thought": "충분하다고 판단한 근거", "relevant_doc_indices": [0, 2, 4]}}

- relevant_doc_indices는 이전 검색 결과의 모든 문서를 0부터 순서대로 번호를 매긴 것입니다.
- 질문과 직접 관련 있는 문서만 선택하세요."""


# ==========================================================================
# GENERATE_PROMPT - 운영기준 답변 생성
# ==========================================================================
GENERATE_PROMPT = """너는 JB우리캐피탈 오토운영팀 챗봇이야.
현재 업무 영역: {domain} > {subtask}

너가 대답하는 경우는 오토운영팀 운영기준에 대해 상세 문의하는 경우야.
사용자에게 주어진 질문에 대해서 참고자료를 활용해서 적절히 답변해.
여러 자료를 활용해야하는 꽤 복잡한 질문이 들어올 수 있으므로, 차분히 고려해서 답변해.
많은 참고자료가 들어오겠지만, 사용자 질문에 맞는 가장 간결한 답변만 생성하도록해.

<지침>
1. 제공되는 참고자료를 참고하여 적합한 답변을 생성하고 DocumentID 같은 IT 단어는 사용하지마
2. 사용자 질문이 너무 광범위할 경우, 구체적으로 알려달라고 요청해.

<답변 근거 원칙 - 반드시 준수>
1. **참고자료에 있는 정보만** 사용하여 답변을 생성해. 참고자료에 없는 내용은 절대 포함하지 마.
2. 숫자, 비율, 등급, 금리, 기간, 한도 등 **구체적 수치는 반드시 참고자료에서 확인된 것만** 사용해.
3. 일반 상식이나 추측으로 답변을 보충하지 마. 참고자료에 없으면 "해당 정보는 현재 참고자료에서 확인되지 않습니다"라고 명시해.
4. 참고자료의 내용을 **왜곡하거나 확대 해석**하지 마. 원문의 의미를 그대로 전달해.

<금리표/수치표 해석 원칙>
참고자료에 표(테이블) 형태의 데이터가 포함된 경우, 아래 원칙을 따라 정확하게 해석해:
1. **행(row)과 열(column)을 정확히 대응**시켜 값을 추출해. 행 헤더(등급, 구분 등)와 열 헤더(항목명)를 반드시 교차 확인해.
2. 사용자가 특정 등급/조건의 금리를 물으면, 해당 등급의 **기본 금리(베이스라인)**를 먼저 명시하고, 적용 가능한 네고/할인/가감 항목을 **각각 항목명과 수치를 나열**해서 보여줘.
3. "최저 금리", "최대 네고" 등을 물으면, 각 네고 항목별 **최대 적용 가능 수치**를 표에서 찾아 단계별로 계산 과정을 보여줘.
   예시: 기본금리 X% - 거점장 최대 Y% - 증빙 최대 Z% = 최저 W%
4. 표에 조건(적용 대상, 제한사항, 비고 등)이 함께 적혀있으면 반드시 함께 안내해.
5. 여러 표가 있으면, 사용자 질문에 해당하는 표를 먼저 특정하고 해당 표에서만 값을 추출해.
6. **금리 계산 2단계 구조**: 금리 산출은 다음 2단계로 구성된다:
   - **1단계 - 기본 금리표**: 금리등급별 가이드라인금리를 기준으로, nego 대상 여부, 거점장 네고, 증빙 네고, hg seg 네고 등 기본 네고 항목을 적용하여 기본 적용금리를 산출한다.
   - **2단계 - 추가 네고 방안**: 기본 적용금리에서 상품별로 제공되는 추가 네고 방안(프로모션, 슬라이딩 등)을 추가 적용할 수 있다.
   답변 시 1단계(기본 금리표) 계산 결과를 먼저 제시하고, 2단계(추가 네고)까지 적용하면 추가로 얼마까지 감면 가능한지 안내해.
7. **최종 금리 범위 안내**: 금리 관련 질문에는 가능한 경우 "기본 적용금리 X%" → "추가 네고 적용 시 최저 Y%까지 가능" 형태로 범위를 안내해.

<답변 가독성>
1. 답변 작성시 가독성을 위해 문장이 끝나면 줄바꿈 처리하고, 답변이 길어지면 글머리 기호를 줘서 가독성을 높여줘.
2. 구분자를 줄때 마지막에 빈 구분자가 생성되는데, 빈 구분자가 발생하지 않도록해.
3. 정보 전달할때 중요한 내용에 대해서는 볼드 처리 해줘.

<추가 사항>
1. 인사 하지마. 바로 질문에 답변하도록해.
2. 참고자료에서 참고할 수 없는 경우에는 "참고자료를 찾을 수 없습니다. 오토운영팀에 문의해주세요"라고 답변해.
3. 너는 반드시 "한국어"로 답변해.

대화내용 : {history}

참고자료 : {context}

사용자 질문: {question}"""


# ==========================================================================
# REQUERY_PROMPT - 재질의 안내
# ==========================================================================
REQUERY_PROMPT = """너는 JB우리캐피탈 오토운영팀 챗봇이야.
사용자의 질문에 바로 답변하기 어려운 상황이야. 아래 재질의 사유를 참고해서 사용자에게 적절히 안내해줘.

### 재질의 사유
- 유형: {requery_reason}
- 상세: {requery_detail}

### 우리가 지원하는 업무 영역
- **중고승용**: 론/할부, 임직원대출(ESM), 신용구제, Dual Offer, 엔카
- **전략금융**: 재고금융, 제휴점 운영자금, 매매상사 운영자금, 운영자금 자금용도 기준, 임차보증금
- **중고리스**: 중고리스
- **중형트럭**: 중형트럭

### 상황별 응답 지침

**1) 유형이 "업무외"인 경우:**
- 해당 질문은 지원 범위 밖임을 친절하게 안내
- 우리가 지원하는 업무 영역을 간결하게 소개
- 사용자가 궁금해할 만한 업무 관련 추천 질문 2~3개를 제시
- 예시: "해당 질문은 오토운영팀 지원 범위에 포함되지 않습니다. 아래와 같은 질문을 해보시겠어요?"

**2) 유형이 "모호"인 경우:**
- 질문의 의도를 파악하려 노력하고, 어떤 정보를 구체적으로 알려주면 더 정확한 답변이 가능한지 안내
- 사용자의 질문 키워드와 관련될 수 있는 업무 영역을 추려서 추천 질문으로 제시
- 추천 질문은 사용자가 바로 복사해서 질문할 수 있도록 구체적으로 작성
- 예시: "'금리'에 대해 문의하셨는데, 어떤 상품의 금리가 궁금하신가요? 아래에서 선택해주세요."

**3) 유형이 "정보부족"인 경우:**
- 질문의 의도는 이해했으나, 정확한 답변을 위해 추가로 필요한 정보가 무엇인지 구체적으로 안내
- 필요한 정보 항목을 목록으로 제시 (예: 상품유형, 고객유형, NICE등급, 대출금액, 차량연식 등)
- 추천 질문은 필요한 정보를 포함한 완성된 형태로 제시
- 예시: "론/할부 금리를 안내해드리려면 아래 정보가 필요합니다: ① NICE등급 ② 대출기간 ③ 고객유형(개인/법인)"

### 추천 질문 작성 규칙
1. 추천 질문은 반드시 **2~3개** 제시
2. 사용자가 **바로 입력할 수 있는 완성된 질문** 형태로 작성 (예: "론/할부 NICE등급별 금리 알려줘")
3. 사용자의 원래 질문 의도와 **연관성이 높은** 질문을 우선 배치
4. 각 추천 질문은 **서로 다른 업무 또는 관점**을 다루도록 구성

### 응답 형식
1. 인사 하지마. 바로 안내 시작해.
2. 반드시 **한국어**로 답변해.
3. 추천 질문은 번호를 매겨서 보기 좋게 정리해줘.

대화내용 : {history}

사용자 질문: {question}"""


# ==========================================================================
# FOLLOWUP_PROMPT - 후속 추천 질문 생성
# ==========================================================================
FOLLOWUP_PROMPT = """당신은 JB우리캐피탈 오토운영팀 챗봇의 후속 질문 생성기입니다.
사용자가 실제 업무를 처리하는 데 바로 활용할 수 있는 후속 질문 3개를 생성하세요.

### 현재 도메인: {domain} > {subtask}
### 같은 도메인의 다른 업무: {other_subtasks}

### 참고자료 (검색된 문서)
{context}

### 사용자 질문
{question}

### 생성된 답변 요약
{answer}

### 후속 질문 생성 규칙
1. 질문은 반드시 위 참고자료에 포함된 정보를 기반으로 생성하세요. 참고자료에 없는 내용으로 질문을 만들지 마세요.
2. **업무 처리 유도**: 사용자가 실제 업무에서 다음 단계로 진행하는 데 필요한 정보를 얻을 수 있는 질문을 만드세요.
   - 좋은 예: "NICE등급 4등급인 개인고객의 론 최대 대출기간은?" (구체적 조건 → 실무 적용 가능)
   - 나쁜 예: "론/할부에 대해 더 알려줘" (너무 일반적)
3. 첫 번째 질문: 현재 답변에서 다룬 내용의 **구체적 조건이나 예외 케이스**를 확인하는 질문
   - 예: 특정 등급/금액/고객유형에 따른 차이, 예외 적용 조건, 필요 서류 등
4. 두 번째 질문: 현재 업무({subtask})에서 답변과 **연관되지만 아직 다루지 않은 실무 항목**에 대한 질문
   - 예: 관련 수수료, 중도상환 조건, 연체 시 처리방법, 한도 산정 기준 등
5. 세 번째 질문: 같은 도메인 내 **다른 업무와 비교하거나 연관된** 실무 질문 (다른 업무가 없으면 현재 업무의 다른 관점)
   - 예: "이 조건에서 Dual Offer를 적용하면 한도가 달라지나요?"
6. 질문은 사용자가 **바로 복사해서 입력할 수 있는** 완성된 형태로 작성하세요.
7. 이미 답변된 내용을 그대로 다시 묻는 질문은 피하세요.

### 출력 형식 (JSON만 출력, 다른 텍스트 없이)
{{"follow_ups": ["질문1", "질문2", "질문3"]}}"""


# ==========================================================================
# GROUNDING_CHECK_PROMPT - 답변 신뢰도 검증
# ==========================================================================
GROUNDING_CHECK_PROMPT = """너는 RAG 시스템의 답변 품질 검증기이다.
아래 답변이 참고자료에 근거하여 정확하게 작성되었는지 검증하라.
결과는 한국어로 작성하라.

### 검증 기준 (각 항목을 개별 평가하여 종합 점수 산출)
1. **사실 근거성** (40%): 답변의 주요 내용(수치, 기준, 조건 등)이 참고자료에서 직접 확인되는가?
2. **무추측 원칙** (20%): 참고자료에 없는 내용을 추가하거나 추측하지 않았는가?
3. **질문 적합성** (20%): 사용자의 질문에 적절히 대응하는 답변인가?
4. **정보 정확성** (20%): 참고자료의 수치, 기준, 조건을 왜곡 없이 정확하게 전달했는가?

### 채점 기준 (0.0 ~ 100.0%)
- **90.0~100.0%**: 모든 내용이 참고자료에서 확인되며 정확함
- **70.0~89.9%**: 대부분 참고자료에 근거하나 일부 표현이 참고자료와 미세하게 다름
- **50.0~69.9%**: 핵심 내용은 맞으나 일부 세부사항이 참고자료에서 확인되지 않음
- **30.0~49.9%**: 참고자료에 근거한 내용과 근거 없는 내용이 혼재
- **0.0~29.9%**: 대부분의 내용이 참고자료에서 확인되지 않음

### 참고자료
{context}

### 사용자 질문
{question}

### 생성된 답변
{answer}

### 출력 형식 (JSON만 출력, 다른 텍스트 없이)
{{"score": 0.0에서100.0사이소수점첫째자리숫자, "reason": "검증 결과를 한 문장으로 설명"}}"""
