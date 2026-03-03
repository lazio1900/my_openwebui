############################################
# 작성자 : 이민재 / Claude
# 생성일자 : 2026-02-24
# 이력
# 2026-02-24 v1 : 멀티 에이전트 통합 Pipe
#   - 기존 5개 파일(all_chat + 4개 도메인) → 단일 파일 통합
#   - 2단계 분류 → 1단계 통합 분류 (LLM 호출 8~9회 → 3회)
#   - 문서 평가 일괄 처리 (개별 5회 → 1회)
#   - 순수 Python async 상태 머신 (추가 패키지 없음)
#   - 개발환경: OpenAI API / 운영환경: generate_chat_completion 전환 가능
# 2026-02-24 v2 : Orchestrator + ReAct 에이전트 전환
#   - 고정 파이프라인(classify→retrieve→grade→generate)을 ReAct 패턴으로 전환
#   - ReAct 루프: Think→Act(vector_search)→Observe 반복 (최대 3회)
#   - 별도 grade 단계 제거 → ReAct 사고 과정에서 문서 관련성 평가
#   - 에이전트가 검색 쿼리 자율 판단/정제 → 검색 품질 향상
#   - LLM 호출: 3~4회 (분류1 + ReAct1~2 + 생성1)
# 2026-02-27 v3 : 후속 질문 생성 (Follow-up Question Generation)
#   - _generate_followups(): RAG 문서 + 도메인 지식 기반 추천 질문 3개 생성
#   - 스트리밍 답변 완료 후 마크다운 형태로 추천 질문 표시
#   - LLM 호출: 4~5회 (분류1 + ReAct1~2 + 생성1 + 후속질문1)
#   - ENABLE_FOLLOWUPS Valve로 기능 ON/OFF 가능
############################################

##########################################################################
# 라이브러리 임포트
##########################################################################
from pydantic import BaseModel          # Open WebUI Valves 설정 모델
from typing import Dict, Callable, Awaitable, Optional, Protocol
import re                               # JSON 파싱 시 코드블록 제거용
import time                             # 실행 시간 측정
import json                             # LLM 응답 JSON 파싱
import asyncio                          # 비동기 상태 Lock
import sys                              # 에러 발생 시 라인 번호 추출

import chromadb                         # ChromaDB 벡터 DB 클라이언트
from langchain_chroma import Chroma     # LangChain-ChromaDB 래퍼 (similarity_search)
from langchain_openai import OpenAIEmbeddings  # OpenAI 임베딩 (text-embedding-3-small)
from openai import AsyncOpenAI          # OpenAI API 비동기 클라이언트

from open_webui.utils.misc import get_last_user_message  # 대화에서 마지막 사용자 메시지 추출
from fastapi import Request             # Open WebUI 요청 객체


##########################################################################
# 비동기 이벤트 통신 유틸리티
# -----------------------------------------------------------------------
# Open WebUI는 __event_emitter__ 콜백을 통해 프론트엔드에 실시간 이벤트 전달.
# - "status" 이벤트: 채팅 UI 상단에 진행 상태 표시 (예: "질문 분류 중...")
# - "citation" 이벤트: 답변 하단에 출처 문서 표시
##########################################################################
EmitterType = Optional[Callable[[dict], Awaitable[None]]]


class SendStatusType(Protocol):
    def __call__(self, status_message: str, done: bool) -> Awaitable[None]: ...


class SendCitationType(Protocol):
    def __call__(self, url: str, title: str, content: str) -> Awaitable[None]: ...


def get_send_status(__event_emitter__: EmitterType):
    """상태 메시지 전송 함수 생성. done=True이면 상태바가 완료 표시로 바뀜."""
    async def send_status(status_message: str, done: bool):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {"type": "status", "data": {"description": status_message, "done": done}}
        )

    return send_status


def get_send_citation(__event_emitter__: EmitterType):
    """출처(citation) 전송 함수 생성. 답변 하단에 참고자료 카드로 표시됨."""
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
# 도메인/서브태스크 매핑 (한 곳에서 관리)
# -----------------------------------------------------------------------
# 4개 도메인, 총 12개 서브태스크로 구성.
# - subtasks: ChromaDB 검색 시 metadata 필터로 사용
# - display_name: UI에 표시되는 이름
# - has_image: 첫 진입 시 이미지 포함 여부 (중고승용만 True)

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

# subtask → domain 역매핑 (예: "(론/할부)" → "중고승용")
SUBTASK_TO_DOMAIN = {}
for domain, info in DOMAIN_SUBTASK_MAP.items():
    for st in info["subtasks"]:
        SUBTASK_TO_DOMAIN[st] = domain


##########################################################################
# Pipe 클래스 (Open WebUI Function 프레임워크)
# -----------------------------------------------------------------------
# Open WebUI에서 Pipe Function은 독립 실행 가능한 챗봇 단위.
# - Valves: 관리자 UI에서 설정 가능한 파라미터 (API 키, 포트 등)
# - pipe(): 사용자 메시지가 들어올 때마다 호출되는 진입점
# - Valves 저장 시 Pipe 인스턴스가 재생성됨 (__init__ 다시 실행)
##########################################################################


class Pipe:
    # ------------------------------------------------------------------
    # Valves: Open WebUI 관리자 화면에서 설정 가능한 파라미터
    # 저장하면 Pipe가 재생성되어 __init__이 다시 실행됨
    # ------------------------------------------------------------------
    class Valves(BaseModel):
        BASE_IMG_URL: str = "https://ai.wooricap.com/static/auto_oper_images/"
        CHROMA_PORT: int = 8800           # ChromaDB 포트 (운영: 8800, Docker 로컬: 8800)
        CHROMA_IP: str = "localhost"      # ChromaDB 호스트 (운영: 172.18.237.81)
        STANDARD_COLLECTION_NAME_IMAGE: str = "auto_oper_standard_image"
        STANDARD_COLLECTION_NAME_TEXT: str = "auto_oper_standard_text"
        EMBEDDING_K: int = 5             # 벡터 검색 시 반환할 문서 수
        # OpenAI API 설정 (개발환경: OpenAI 직접 / 운영환경: generate_chat_completion)
        OPENAI_API_KEY: str = ""         # Valves UI에서 입력
        OPENAI_MODEL: str = "gpt-4o-mini"
        OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
        # ReAct 에이전트 설정
        REACT_MAX_ITERATIONS: int = 3    # Think→Act→Observe 최대 반복 횟수
        # 후속 질문 생성 설정
        ENABLE_FOLLOWUPS: bool = True    # False로 하면 추천 질문 미표시

    def __init__(self):
        self.valves = self.Valves()
        self._state_by_user: Dict[str, dict] = {}  # 사용자별 멀티턴 상태 저장소
        self._state_lock = asyncio.Lock()           # 동시 접근 방지 Lock
        self._state_ttl_sec = 60 * 60               # 1시간 후 상태 자동 삭제

        # ★ Lazy 초기화: __init__ 시점에는 Valves가 기본값(빈 API 키)이므로,
        #   실제 pipe() 호출 시점에 클라이언트를 생성해야 Valves 값이 반영됨
        self.openai_client = None
        self.chroma_db_text = None
        self.chroma_db_image = None
        self._initialized = False

        print("Orchestrator + ReAct 에이전트 Pipe 등록 완료! (lazy init)")

    def _ensure_initialized(self):
        """pipe() 첫 호출 시 실행. Valves 값이 확정된 후 외부 클라이언트 초기화."""
        if self._initialized:
            return

        # OpenAI 비동기 클라이언트 (LLM 호출용)
        self.openai_client = AsyncOpenAI(api_key=self.valves.OPENAI_API_KEY)

        # ChromaDB HTTP 클라이언트 (벡터 DB 연결)
        chroma_client = chromadb.HttpClient(
            host=self.valves.CHROMA_IP, port=self.valves.CHROMA_PORT
        )

        # 임베딩 함수 (ChromaDB 검색 시 쿼리를 벡터로 변환)
        embedding_func = OpenAIEmbeddings(
            model=self.valves.OPENAI_EMBEDDING_MODEL,
            openai_api_key=self.valves.OPENAI_API_KEY,
        )

        # Chroma 검색 객체 2개 (텍스트 컬렉션 + 이미지 컬렉션)
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

        self._initialized = True
        print("Orchestrator + ReAct 에이전트 초기화 완료!")

    # =================================================================
    # pipe() - Open WebUI가 사용자 메시지마다 호출하는 메인 엔트리포인트
    # ---------------------------------------------------------------
    # 호출 흐름:
    #   1. _classify()      : 사용자 질문을 도메인/서브태스크로 분류 (LLM 1회)
    #   2. _requery()       : 분류 불가 시 재질의 응답 반환 (LLM 1회)
    #   3. _react_agent()   : 벡터 검색으로 관련 문서 수집 (LLM 1~2회)
    #   4. _generate()      : 수집된 문서 기반 스트리밍 답변 생성 (LLM 1회)
    #      + _generate_followups() : 후속 추천 질문 생성 (LLM 1회)
    #
    # 반환값:
    #   - AsyncGenerator (스트리밍) 또는 Generator (이미지 포함 시)
    #   - Open WebUI가 generator를 소비하여 프론트엔드에 실시간 전송
    # =================================================================
    async def pipe(
        self,
        body: dict,                        # Open WebUI가 전달하는 요청 body (messages, user 등)
        __event_emitter__: Callable[[dict], Awaitable[None]] | None = None,  # 프론트엔드 이벤트 콜백
        __user__: dict = None,             # 현재 로그인 사용자 정보
        __request__: Request = None,       # FastAPI Request 객체
    ):
        try:
            # ★ Lazy 초기화: 첫 호출 시에만 OpenAI/ChromaDB 클라이언트 생성
            self._ensure_initialized()
            start_time = time.time()
            user_id = self._get_user_id(body, __user__)
            user_message = get_last_user_message(body["messages"])  # 마지막 사용자 메시지만 추출
            messages = body["messages"]       # 전체 대화 히스토리 (system, user, assistant 메시지)
            send_status = get_send_status(__event_emitter__)  # 상태 표시 유틸리티

            # ── 멀티턴 상태 관리 ──
            # 사용자별로 이전 분류 결과(도메인, 서브태스크)를 메모리에 보관.
            # TTL(1시간)이 지난 상태는 자동 삭제하여 메모리 누수 방지.
            # 이전 분류 결과는 _classify()에서 멀티턴 연속성 판단에 사용.
            now = time.time()
            async with self._state_lock:     # 동시 요청 시 race condition 방지
                # TTL 만료된 사용자 상태 정리
                for k, v in list(self._state_by_user.items()):
                    if now - v.get("ts", 0) > self._state_ttl_sec:
                        del self._state_by_user[k]
                # 현재 사용자의 이전 상태 로드
                user_state = self._state_by_user.get(user_id, {})
                last_classified = user_state.get("last_classified", "")  # 이전 서브태스크
                last_domain = user_state.get("last_domain", "")          # 이전 도메인

            # ── 파이프라인 상태 객체 초기화 ──
            # 모든 단계(_classify, _react_agent, _generate 등)가 이 state dict를 공유.
            # 각 단계에서 결과를 state에 기록하고 다음 단계가 읽어서 사용.
            state = {
                # 입력 데이터
                "user_message": user_message,     # 현재 사용자 질문 텍스트
                "messages": messages,             # 전체 대화 히스토리
                "user_id": user_id,               # 사용자 식별자
                "event_emitter": __event_emitter__,  # 이벤트 콜백 (citation 전송용)
                "send_status": send_status,       # 상태바 업데이트 함수
                "start_time": start_time,         # 실행 시간 측정 시작점
                # _classify()가 채우는 분류 결과
                "domain": "",                     # 도메인명 (예: "중고승용")
                "subtask": "",                    # 서브태스크명 (예: "(론/할부)")
                "is_requery": False,              # 재질의 여부 (분류 불가 시 True)
                # _react_agent()가 채우는 검색 결과
                "filtered_docs": [],              # 관련성 높은 문서 리스트
                # 멀티턴 판단용 이전 상태
                "is_first_entry": False,          # 해당 서브태스크 첫 진입 여부
                "last_classified": last_classified,  # 직전 턴의 서브태스크
                "last_domain": last_domain,       # 직전 턴의 도메인
            }

            # ── 파이프라인 실행 (4단계) ──

            # 1단계: Orchestrator - 사용자 질문을 도메인/서브태스크로 분류
            state = await self._classify(state)

            # 2단계: 재질의 분기 - 분류 불가 시 안내 메시지 반환 후 종료
            if state["is_requery"]:
                return await self._requery(state)

            # 3단계: ReAct 에이전트 - Think→Act(검색)→Observe 루프로 문서 수집
            state = await self._react_agent(state)

            # 4단계: 최종 답변 생성 - 수집된 문서 기반 스트리밍 답변 + 후속 질문
            return await self._generate(state)

        except Exception as e:
            # ── 전역 에러 핸들링 ──
            # 어떤 단계에서든 예외 발생 시 라인 번호 포함 에러 메시지를 프론트엔드에 표시.
            # 사용자에게는 에러 상태바 + 에러 텍스트를 반환.
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"exception :: {e}\n라인: {exc_tb.tb_lineno}"
            send_status = get_send_status(__event_emitter__)
            await send_status(status_message=f"오류 발생: {e}", done=True)
            return f"\n\n#### [Error]\n{error_message}\n"

    # =================================================================
    # _classify() - Orchestrator: 사용자 질문 → 도메인/서브태스크 분류
    # ---------------------------------------------------------------
    # LLM 1회 호출로 4개 도메인 × 12개 서브태스크 중 하나로 분류.
    # 멀티턴 대화에서는 이전 분류 결과(last_domain, last_classified)를
    # 프롬프트에 포함하여 맥락 연속성을 유지.
    # 분류 불가 시 domain="재질의"로 설정 → pipe()에서 _requery()로 분기.
    #
    # state 업데이트: domain, subtask, is_requery, is_first_entry
    # =================================================================
    async def _classify(self, state: dict) -> dict:
        send_status = state["send_status"]
        await send_status(status_message="질문 분류 중 ...", done=False)

        # 최근 대화 히스토리를 텍스트로 요약 (멀티턴 맥락 파악용)
        history_summary = self._build_history_summary(state["messages"])

        # CLASSIFY_PROMPT에 변수 주입하여 분류 프롬프트 생성
        # - history: 최근 대화 요약 (멀티턴 맥락)
        # - last_domain/last_subtask: 직전 턴의 분류 결과 (연속성 판단)
        # - question: 현재 사용자 질문
        prompt = CLASSIFY_PROMPT.format(
            history=history_summary,
            last_domain=state["last_domain"] or "없음",
            last_subtask=state["last_classified"] or "없음",
            question=state["user_message"],
        )

        # LLM 호출 → JSON 응답 수신 (예: {"domain":"중고승용","subtask":"(론/할부)"})
        content = await self._llm_call(prompt)

        # JSON에서 domain, subtask 추출 + 유효성 검증
        # (파싱 실패 시 텍스트에서 서브태스크 키워드를 탐색하는 폴백 로직 포함)
        domain, subtask = self._parse_classification(content)

        # 첫 진입 여부 판단:
        # - 이전 분류 결과가 없거나 서브태스크가 변경되면 첫 진입
        # - 첫 진입 시 이미지 포함 답변 생성 (중고승용 도메인만)
        is_first_entry = (
            state["last_classified"] == ""
            or state["last_classified"] != subtask
        )

        # state에 분류 결과 기록
        state["domain"] = domain
        state["subtask"] = subtask
        state["is_requery"] = domain == "재질의"   # 분류 불가 → 재질의 경로
        state["is_first_entry"] = is_first_entry

        if not state["is_requery"]:
            # 분류 성공 시: 프론트엔드 상태바에 결과 표시 + 사용자 상태 저장
            display = DOMAIN_SUBTASK_MAP.get(domain, {}).get(
                "display_name", domain
            )
            await send_status(
                status_message=f"분류 결과 : {display} > {subtask}", done=True
            )
            # 다음 턴의 멀티턴 연속성을 위해 분류 결과를 메모리에 저장
            await self._update_state(state["user_id"], domain, subtask)

        return state

    # =================================================================
    # _react_agent() - ReAct 패턴 루프 컨트롤러
    # ---------------------------------------------------------------
    # ReAct(Reasoning + Acting) 패턴: LLM이 스스로 "생각(Think)"하고,
    # "행동(Act=벡터검색)"을 수행하고, "관찰(Observe=검색결과)"한 뒤,
    # 충분한 정보가 모이면 "완료(FINISH)"를 선언하는 자율 루프.
    #
    # 기존 고정 파이프라인(검색→평가→생성)과 달리:
    # - 에이전트가 검색 쿼리를 자율적으로 생성/정제
    # - 검색 결과 부족 시 자동으로 추가 검색 (다른 키워드/서브태스크)
    # - 관련 문서만 선별하여 filtered_docs에 저장
    #
    # 최대 반복: REACT_MAX_ITERATIONS (기본 3회)
    # state 업데이트: filtered_docs
    # =================================================================
    async def _react_agent(self, state: dict) -> dict:
        """ReAct 루프: Think→Act→Observe 반복으로 관련 문서 수집"""
        # observations: 각 검색 반복의 결과를 누적 저장
        # 구조: [{"query": 검색어, "subtask": 대상, "docs": 원본문서, "doc_summaries": 요약}, ...]
        observations = []
        send_status = state["send_status"]

        for i in range(self.valves.REACT_MAX_ITERATIONS):
            await send_status(
                status_message=f"ReAct 에이전트 실행 중... (반복 {i + 1}/{self.valves.REACT_MAX_ITERATIONS})",
                done=False,
            )

            # ── Think: LLM이 다음 행동을 결정 ──
            # observations(이전 검색 결과)를 보고 SEARCH 또는 FINISH 판단
            decision = await self._react_step(state, observations)

            if decision["action"] == "FINISH":
                # ── FINISH: 에이전트가 충분한 정보를 수집했다고 판단 ──
                # LLM이 선택한 관련 문서 인덱스(relevant_doc_indices)로 필터링
                relevant_indices = decision.get("relevant_doc_indices", [])
                all_docs = []
                for obs in observations:
                    all_docs.extend(obs["docs"])

                # 에이전트가 지정한 인덱스의 문서만 선별
                if relevant_indices and all_docs:
                    state["filtered_docs"] = [
                        all_docs[idx] for idx in relevant_indices
                        if 0 <= idx < len(all_docs)  # 범위 초과 방지
                    ]
                else:
                    # 인덱스가 없으면 전체 문서 사용
                    state["filtered_docs"] = all_docs

                # 안전장치: 필터링 후 0건이면 전체 문서로 폴백
                if not state["filtered_docs"] and all_docs:
                    state["filtered_docs"] = all_docs
                break

            elif decision["action"] == "SEARCH":
                # ── Act: ChromaDB 벡터 검색 실행 ──
                query = decision["query"]      # LLM이 생성한 검색 쿼리
                subtask = decision["subtask"]  # LLM이 선택한 대상 서브태스크

                await send_status(
                    status_message=f"{subtask} 검색 중: \"{query[:30]}...\"" if len(query) > 30 else f"{subtask} 검색 중: \"{query}\"",
                    done=False,
                )

                # 실제 벡터 검색 수행 (LLM 호출 없음, 순수 DB 쿼리)
                docs = self._execute_tool(query, subtask)

                # ── Observe: 검색 결과를 관측 기록에 추가 ──
                # offset: 이전 검색에서 이미 쌓인 문서 수 (전체 인덱스 매핑용)
                # 예: 1차 검색에서 5개 → 2차 검색 문서는 [문서5], [문서6], ... 으로 번호 매김
                offset = sum(len(obs["docs"]) for obs in observations)

                observations.append({
                    "query": query,
                    "subtask": subtask,
                    "docs": docs,                    # 원본 Document 객체 (답변 생성에 사용)
                    "doc_summaries": [               # 요약 텍스트 (다음 _react_step에 전달)
                        f"[문서{offset + j}] {d.page_content[:200]}"
                        for j, d in enumerate(docs)
                    ],
                })
        else:
            # for-else: 최대 반복까지 FINISH 없이 루프 종료 시 → 전체 문서 사용
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
    # _react_step() - ReAct 1스텝: LLM이 다음 행동 결정
    # ---------------------------------------------------------------
    # LLM에게 현재까지의 검색 결과(observations)와 사용자 질문을 보여주고,
    # "추가 검색이 필요한가?"를 판단하게 함.
    #
    # 반환값 예시:
    #   {"action": "SEARCH", "query": "론/할부 금리등급", "subtask": "(론/할부)"}
    #   {"action": "FINISH", "relevant_doc_indices": [0, 2, 4]}
    # =================================================================
    async def _react_step(self, state: dict, observations: list) -> dict:
        """1회 LLM 호출: 다음 행동 결정 (SEARCH or FINISH)"""
        domain = state["domain"]
        subtask = state["subtask"]
        domain_info = DOMAIN_SUBTASK_MAP.get(domain, {})
        # 현재 도메인에서 사용 가능한 모든 서브태스크 (교차 검색 허용)
        available_subtasks = ", ".join(domain_info.get("subtasks", [subtask]))

        # 이전 검색 결과를 텍스트로 변환 (LLM에게 전달)
        # 각 검색의 쿼리/서브태스크 + 문서 요약(200자)을 포함
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

        content = await self._llm_call(prompt)
        # JSON 응답을 action dict로 변환 (파싱 실패 시 기본 검색 수행)
        return self._parse_react_decision(content, state)

    # =================================================================
    # _execute_tool() - 벡터 검색 도구 실행
    # ---------------------------------------------------------------
    # ChromaDB에서 similarity_search 수행 (코사인 유사도).
    # metadata의 subtask 필드로 필터링하여 해당 서브태스크 문서만 검색.
    # EMBEDDING_K(기본 5)개의 가장 유사한 문서를 반환.
    # =================================================================
    def _execute_tool(self, query: str, subtask: str) -> list:
        """ChromaDB 벡터 검색 (LLM 호출 없음, 순수 DB 쿼리)"""
        return self.chroma_db_text.similarity_search(
            query=query,                          # 검색 쿼리 (임베딩으로 변환됨)
            k=self.valves.EMBEDDING_K,            # 반환할 문서 수
            filter={"subtask": subtask},          # metadata 필터 (서브태스크별 검색)
        )

    # =================================================================
    # _generate() - 최종 답변 생성 (스트리밍)
    # ---------------------------------------------------------------
    # ReAct 에이전트가 수집한 문서(filtered_docs)를 참고자료로 활용하여
    # 사용자 질문에 대한 최종 답변을 OpenAI 스트리밍으로 생성.
    #
    # 출력 구조 (순서대로 yield):
    #   1. prefix: "#### [중고승용 운영기준 > (론/할부)]"
    #   2. LLM 스트리밍 답변 (토큰 단위로 실시간 전송)
    #   3. suffix: "**[중고승용 운영기준입니다. 다른 운영기준이 필요한 경우...]**"
    #   4. 후속 추천 질문 (ENABLE_FOLLOWUPS=True일 때)
    #
    # 반환값: AsyncGenerator (Open WebUI가 소비하여 프론트엔드에 전달)
    #
    # 분기: 첫 진입 + 이미지 도메인(중고승용) → _generate_with_image()로 위임
    # =================================================================
    async def _generate(self, state: dict):
        send_status = state["send_status"]
        domain = state["domain"]
        subtask = state["subtask"]
        filtered_docs = state["filtered_docs"]
        user_message = state["user_message"]
        messages = state["messages"]
        display_name = DOMAIN_SUBTASK_MAP.get(domain, {}).get(
            "display_name", domain
        )

        # ── 이미지 경로 분기 ──
        # 해당 서브태스크 첫 진입 + 이미지가 있는 도메인(중고승용) → 이미지 포함 답변
        if state["is_first_entry"] and DOMAIN_SUBTASK_MAP.get(domain, {}).get("has_image", False):
            return await self._generate_with_image(state)

        # ── 참고자료 컨텍스트 구성 ──
        # filtered_docs의 page_content를 [참고자료 1], [참고자료 2]... 형태로 연결
        context = "\n\n".join(
            [f"[참고자료 {i+1}]\n{doc.page_content}" for i, doc in enumerate(filtered_docs)]
        )

        # GENERATE_PROMPT에 변수 주입 (도메인, 서브태스크, 대화이력, 참고자료, 질문)
        prompt = GENERATE_PROMPT.format(
            domain=display_name,
            subtask=subtask,
            history=str(messages[-6:]) if len(messages) > 6 else str(messages),
            context=context,
            question=user_message,
        )

        # 실행 시간 표시
        end_time = time.time()
        exe_time = end_time - state["start_time"]
        await send_status(status_message=f"완료: {exe_time:.2f}초", done=True)

        # ── 출처(citation) 전송 ──
        # 각 참고자료를 citation 이벤트로 전송 → 프론트엔드 답변 하단에 출처 카드 표시
        send_citation = get_send_citation(state["event_emitter"])
        for idx, doc in enumerate(filtered_docs, start=1):
            await send_citation(
                url=f"출처{idx}",
                title=f"출처{idx}",
                content=doc.page_content,
            )

        # ── AsyncGenerator 구성: 스트리밍 응답 ──
        # Open WebUI는 이 generator를 소비하여 각 토큰을 프론트엔드에 SSE로 전송.
        prefix = f"#### [{display_name} > {subtask}]\n"
        suffix = f"\n\n **[{display_name}입니다. 다른 운영기준이 필요한 경우 '새 채팅'을 이용해주세요.]**"

        async def stream_response():
            # 1) 헤더 출력 (도메인 > 서브태스크)
            yield prefix

            # 2) OpenAI 스트리밍 답변 (토큰 단위 실시간 전송)
            collected_answer = []  # 후속 질문 생성을 위해 답변 텍스트 캡처
            stream = await self.openai_client.chat.completions.create(
                model=self.valves.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0,    # 결정론적 출력 (운영기준 안내이므로 창의성 불필요)
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    collected_answer.append(delta.content)
                    yield delta.content   # 프론트엔드에 토큰 실시간 전송

            # 3) 안내 문구 출력
            yield suffix

            # 4) 후속 추천 질문 생성 (v3 기능)
            # 스트리밍 완료 후 1회 추가 LLM 호출로 추천 질문 3개 생성
            if self.valves.ENABLE_FOLLOWUPS:
                full_answer = "".join(collected_answer)
                follow_ups = await self._generate_followups(state, full_answer)
                if follow_ups:
                    followup_text = "\n\n---\n**추천 질문:**\n"
                    for idx, q in enumerate(follow_ups, 1):
                        followup_text += f"{idx}. {q}\n"
                    yield followup_text

        return stream_response()

    # =================================================================
    # _generate_with_image() - 이미지 포함 답변 생성 (non-streaming)
    # ---------------------------------------------------------------
    # 중고승용 도메인 첫 진입 시 호출.
    # ChromaDB 이미지 컬렉션에서 서브태스크에 해당하는 운영기준 이미지를 검색.
    # non-streaming으로 LLM 답변을 생성한 후, 이미지 + 답변 + 후속질문을
    # 하나의 문자열로 결합하여 sync generator로 반환.
    #
    # ★ _generate()와 달리 스트리밍이 아닌 이유:
    #    이미지 URL을 답변 상단에 배치해야 하므로 전체 텍스트를 먼저 조합.
    #    sync generator(char 단위)로 반환하여 Open WebUI의 타이핑 효과 유지.
    # =================================================================
    async def _generate_with_image(self, state: dict):
        """첫 진입시 이미지 + 답변 생성 (non-streaming → sync generator)"""
        domain = state["domain"]
        subtask = state["subtask"]
        filtered_docs = state["filtered_docs"]
        user_message = state["user_message"]
        messages = state["messages"]
        display_name = DOMAIN_SUBTASK_MAP.get(domain, {}).get(
            "display_name", domain
        )

        # ── 이미지 검색 ──
        # ChromaDB 이미지 컬렉션에서 서브태스크에 해당하는 이미지 URL 검색
        # metadata의 html_images 필드에 이미지 URL 저장되어 있음
        image_results = self.chroma_db_image.similarity_search(
            query=subtask, k=1
        )

        image_url = ""
        if image_results:
            image_url = image_results[0].metadata.get("html_images", "")
        if not image_url:
            image_url = "이미지 검색 결과 없음"

        # ── 참고자료 컨텍스트 구성 (텍스트 문서) ──
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
        end_time = time.time()
        exe_time = end_time - state["start_time"]
        await send_status(status_message=f"완료: {exe_time:.2f}초", done=True)

        # non-streaming LLM 호출 (이미지와 조합해야 하므로 전체 텍스트 한번에 생성)
        llm_text = await self._llm_call(prompt)

        # ── 후속 추천 질문 생성 (v3 기능) ──
        followup_text = ""
        if self.valves.ENABLE_FOLLOWUPS:
            follow_ups = await self._generate_followups(state, llm_text)
            if follow_ups:
                followup_text = "\n\n---\n**추천 질문:**\n"
                for idx, q in enumerate(follow_ups, 1):
                    followup_text += f"{idx}. {q}\n"

        # ── 최종 출력 조합: 헤더 + 이미지 + LLM답변 + 후속질문 ──
        combined = f"#### [{display_name} > {subtask}]\n{image_url}\n{llm_text}{followup_text}"

        # sync generator: 문자 단위로 yield하여 타이핑 효과 구현
        def stream_output():
            for char in combined:
                yield char

        return stream_output()

    # =================================================================
    # _generate_followups() - 후속 추천 질문 생성 (v3 기능)
    # ---------------------------------------------------------------
    # 답변 완료 후 1회 추가 LLM 호출로 후속 추천 질문 3개를 생성.
    # 추가 벡터 검색 없이 기존 filtered_docs를 context로 재활용.
    #
    # 생성 전략 (FOLLOWUP_PROMPT에 정의):
    #   1번 질문: 같은 서브태스크 내에서 답변에서 다루지 않은 세부 내용
    #   2번 질문: 현재 답변을 더 깊이 파고드는 질문 (조건, 예외, 수치)
    #   3번 질문: 같은 도메인 내 다른 서브태스크 관련 질문
    #
    # 에러 발생 시 빈 리스트 반환 (답변 품질에 영향 없음).
    # =================================================================
    async def _generate_followups(self, state: dict, answer: str) -> list:
        """후속 질문 3개 생성 (1 LLM call, non-streaming)"""
        try:
            domain = state["domain"]
            subtask = state["subtask"]
            filtered_docs = state.get("filtered_docs", [])
            domain_info = DOMAIN_SUBTASK_MAP.get(domain, {})
            display_name = domain_info.get("display_name", domain)

            # 기존 검색 결과로 context 구성 (추가 벡터 검색 없음)
            context = "\n\n".join(
                [f"[참고자료 {i+1}]\n{doc.page_content}" for i, doc in enumerate(filtered_docs)]
            )

            # 같은 도메인 내 다른 서브태스크 목록 추출
            # 예: 현재 (론/할부) → 다른: (임직원대출), (신용구제), (Dual Offer), (엔카)
            all_subtasks = domain_info.get("subtasks", [])
            other_subtasks = [s for s in all_subtasks if s != subtask]
            other_subtasks_str = ", ".join(other_subtasks) if other_subtasks else "없음"

            prompt = FOLLOWUP_PROMPT.format(
                domain=display_name,
                subtask=subtask,
                other_subtasks=other_subtasks_str,
                context=context,
                question=state["user_message"],
                answer=answer[:2000],  # 답변이 너무 길면 2000자로 잘라서 전달
            )

            content = await self._llm_call(prompt)

            # JSON 응답 파싱: {"follow_ups": ["질문1", "질문2", "질문3"]}
            cleaned = re.sub(r"```(?:json)?", "", content).strip().strip("`")
            data = json.loads(cleaned)
            follow_ups = data.get("follow_ups", [])

            # 최대 3개까지만 반환
            if isinstance(follow_ups, list) and len(follow_ups) >= 1:
                return [str(q) for q in follow_ups[:3]]
            return []

        except Exception as e:
            # 후속 질문 생성 실패 시 빈 리스트 반환 (답변은 이미 완료되었으므로 무시)
            print(f"[Followup generation error] {e}")
            return []

    # =================================================================
    # _requery() - 재질의 응답 생성 (스트리밍)
    # ---------------------------------------------------------------
    # _classify()에서 domain="재질의"로 판단된 경우 호출.
    # RAG 검색 없이 LLM만으로 안내 메시지를 생성하여 사용자에게 반환.
    #
    # 사용 사례:
    #   - "운영기준 알려줘" (범위가 너무 넓음) → 구체적 질문 예시 제공
    #   - "날씨 어때?" (업무 외 질문) → 지원 업무 영역 안내
    #   - 키워드가 모호하여 도메인/서브태스크를 특정할 수 없는 경우
    # =================================================================
    async def _requery(self, state: dict):
        send_status = state["send_status"]
        await send_status(status_message="재질의 응답 생성 중...", done=False)

        # REQUERY_PROMPT: 지원 업무 영역 목록 + 추천 질문 예시를 포함하는 프롬프트
        prompt = REQUERY_PROMPT.format(
            history=str(state["messages"][-6:]),
            question=state["user_message"],
        )

        end_time = time.time()
        exe_time = end_time - state["start_time"]
        await send_status(status_message=f"완료: {exe_time:.2f}초", done=True)

        # OpenAI 스트리밍 → AsyncGenerator로 반환
        async def stream_response():
            stream = await self.openai_client.chat.completions.create(
                model=self.valves.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content

        return stream_response()

    # =================================================================
    # 유틸리티 함수
    # =================================================================

    # -----------------------------------------------------------------
    # _llm_call() - 단일 프롬프트 LLM 호출 (non-streaming)
    # 분류, ReAct, 후속질문 등 JSON 응답이 필요한 곳에서 사용.
    # temperature=0으로 결정론적 출력 보장.
    # -----------------------------------------------------------------
    async def _llm_call(self, prompt: str) -> str:
        """non-stream LLM 호출 (단일 프롬프트) → 텍스트 반환"""
        response = await self.openai_client.chat.completions.create(
            model=self.valves.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0,
        )
        return response.choices[0].message.content

    # -----------------------------------------------------------------
    # _llm_call_messages() - messages 배열 기반 LLM 호출 (non-streaming)
    # system/user/assistant 역할이 구분된 대화형 프롬프트에 사용.
    # 현재 코드에서는 직접 사용되지 않지만, 확장 시 활용 가능.
    # -----------------------------------------------------------------
    async def _llm_call_messages(self, messages: list) -> str:
        """non-stream LLM 호출 (messages 배열) → 텍스트 반환"""
        response = await self.openai_client.chat.completions.create(
            model=self.valves.OPENAI_MODEL,
            messages=messages,
            stream=False,
            temperature=0,
        )
        return response.choices[0].message.content

    # -----------------------------------------------------------------
    # _get_user_id() - 사용자 ID 추출
    # Open WebUI에서 전달하는 __user__ 또는 body에서 사용자 ID를 추출.
    # 멀티턴 상태 관리(_state_by_user)의 키로 사용.
    # -----------------------------------------------------------------
    def _get_user_id(self, body: dict, __user__: dict | None = None) -> str:
        if isinstance(__user__, dict) and __user__.get("id"):
            return __user__["id"]
        return (body.get("user") or {}).get("id") or "anonymous"

    # -----------------------------------------------------------------
    # _update_state() - 사용자별 멀티턴 상태 저장
    # 현재 분류 결과(domain, subtask)를 메모리에 저장.
    # 다음 턴의 _classify()에서 last_domain, last_classified로 참조.
    # asyncio.Lock으로 동시 접근 방지.
    # -----------------------------------------------------------------
    async def _update_state(self, user_id: str, domain: str, subtask: str) -> None:
        async with self._state_lock:
            self._state_by_user[user_id] = {
                "last_classified": subtask,    # 이전 서브태스크 (멀티턴 연속성)
                "last_domain": domain,         # 이전 도메인
                "ts": time.time(),             # TTL 체크용 타임스탬프
            }

    # -----------------------------------------------------------------
    # _build_history_summary() - 대화 히스토리 텍스트 요약
    # 분류 프롬프트에 포함할 최근 대화 내용을 생성.
    # 마지막 메시지(현재 질문)를 제외한 최근 6개 메시지를 사용.
    # 300자 초과 메시지는 앞뒤 150자만 남기고 중간 생략.
    # -----------------------------------------------------------------
    def _build_history_summary(self, messages: list) -> str:
        """최근 대화 히스토리를 요약 텍스트로 변환"""
        if len(messages) <= 1:
            return "없음 (첫 질문)"

        history_parts = []
        recent = messages[-7:-1]  # 최근 6개 메시지 (마지막=현재 질문 제외)
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")
            # 긴 메시지는 앞뒤만 남기고 중략 처리 (프롬프트 토큰 절약)
            if len(content) > 300:
                content = content[:150] + "\n...(중략)...\n" + content[-150:]
            history_parts.append(f"[{role}] {content}")

        return "\n".join(history_parts) if history_parts else "없음 (첫 질문)"

    # -----------------------------------------------------------------
    # _parse_classification() - 분류 LLM 응답 JSON 파싱
    # ---------------------------------------------------------------
    # LLM 응답에서 domain, subtask를 추출하고 유효성 검증.
    #
    # 파싱 전략 (우선순위):
    #   1) JSON 파싱 → DOMAIN_SUBTASK_MAP에서 유효성 검증
    #   2) subtask 불일치 시 → 괄호 제거 후 부분 매칭 시도
    #   3) 서브태스크 1개뿐인 도메인 → 자동 매핑
    #   4) JSON 파싱 실패 → 텍스트에서 서브태스크 키워드 탐색 (폴백)
    #   5) 모든 방법 실패 → ("재질의", "재질의") 반환
    # -----------------------------------------------------------------
    def _parse_classification(self, content: str) -> tuple:
        """분류 LLM 응답에서 domain, subtask 파싱 + 유효성 검증"""
        try:
            # 코드블록(```json ... ```) 제거 후 순수 JSON 추출
            cleaned = re.sub(r"```(?:json)?", "", content).strip().strip("`")
            data = json.loads(cleaned)
            domain = data.get("domain", "재질의")
            subtask = data.get("subtask", "재질의")

            # 도메인이 유효한지 확인
            if domain in DOMAIN_SUBTASK_MAP:
                valid_subtasks = DOMAIN_SUBTASK_MAP[domain]["subtasks"]
                if subtask not in valid_subtasks:
                    # subtask가 정확히 일치하지 않으면 부분 매칭 시도
                    # 예: LLM이 "론/할부"로 응답 → "(론/할부)"로 보정
                    for vs in valid_subtasks:
                        if subtask.replace("(", "").replace(")", "") in vs:
                            subtask = vs
                            break
                    else:
                        # 서브태스크가 1개뿐인 도메인(중고리스, 중형트럭) → 자동 매핑
                        if len(valid_subtasks) == 1:
                            subtask = valid_subtasks[0]
                return domain, subtask
            else:
                # 알 수 없는 도메인 → 재질의
                return "재질의", "재질의"
        except (json.JSONDecodeError, KeyError, AttributeError):
            # JSON 파싱 실패 → 텍스트에서 서브태스크 키워드 직접 탐색 (폴백)
            for domain, info in DOMAIN_SUBTASK_MAP.items():
                for st in info["subtasks"]:
                    if st in content:
                        return domain, st
            return "재질의", "재질의"

    # -----------------------------------------------------------------
    # _parse_react_decision() - ReAct Step LLM 응답 JSON 파싱
    # ---------------------------------------------------------------
    # LLM 응답에서 action(SEARCH/FINISH)을 추출.
    #
    # SEARCH 시: query(검색어), subtask(대상) 추출 + subtask 유효성 검증
    # FINISH 시: relevant_doc_indices(관련 문서 번호) 추출
    # 파싱 실패 시: 기본 검색 수행 (사용자 원문으로 현재 서브태스크 검색)
    # -----------------------------------------------------------------
    def _parse_react_decision(self, content: str, state: dict) -> dict:
        """ReAct Step LLM 응답에서 action dict 파싱"""
        try:
            cleaned = re.sub(r"```(?:json)?", "", content).strip().strip("`")
            data = json.loads(cleaned)
            action = data.get("action", "FINISH")

            if action == "SEARCH":
                query = data.get("query", state["user_message"])
                subtask = data.get("subtask", state["subtask"])

                # subtask 유효성 검증 (LLM이 잘못된 서브태스크를 지정할 수 있음)
                domain_info = DOMAIN_SUBTASK_MAP.get(state["domain"], {})
                valid_subtasks = domain_info.get("subtasks", [])
                if subtask not in valid_subtasks:
                    # 부분 매칭 시도 (괄호 제거 후 비교)
                    for vs in valid_subtasks:
                        if subtask.replace("(", "").replace(")", "") in vs:
                            subtask = vs
                            break
                    else:
                        # 매칭 실패 → 원래 분류된 서브태스크로 폴백
                        subtask = state["subtask"]

                return {"action": "SEARCH", "query": query, "subtask": subtask}

            elif action == "FINISH":
                # 관련 문서 인덱스 추출 (정수만 필터링)
                relevant_indices = data.get("relevant_doc_indices", [])
                return {
                    "action": "FINISH",
                    "relevant_doc_indices": [
                        i for i in relevant_indices if isinstance(i, int)
                    ],
                }

            else:
                # 알 수 없는 action → 안전하게 FINISH로 처리
                return {"action": "FINISH", "relevant_doc_indices": []}

        except (json.JSONDecodeError, KeyError, AttributeError):
            # JSON 파싱 실패 → 사용자 원문으로 기본 검색 수행
            return {
                "action": "SEARCH",
                "query": state["user_message"],
                "subtask": state["subtask"],
            }


###################################################################################
# 프롬프트 템플릿 (Prompt Templates)
# -----------------------------------------------------------------------
# 각 단계에서 LLM에게 전달하는 프롬프트를 상수로 정의.
# Python str.format()으로 변수를 주입하여 사용.
#
# ┌─────────────────┬────────────────────────────────────────────────┐
# │ 프롬프트          │ 사용 위치                                       │
# ├─────────────────┼────────────────────────────────────────────────┤
# │ CLASSIFY_PROMPT │ _classify() - 질문→도메인/서브태스크 분류          │
# │ REACT_STEP_PROMPT│ _react_step() - 검색 or 완료 판단              │
# │ GENERATE_PROMPT │ _generate(), _generate_with_image() - 답변 생성  │
# │ REQUERY_PROMPT  │ _requery() - 분류 불가 시 재질의 안내              │
# │ FOLLOWUP_PROMPT │ _generate_followups() - 추천 질문 3개 생성        │
# └─────────────────┴────────────────────────────────────────────────┘
###################################################################################

# ==========================================================================
# CLASSIFY_PROMPT - 질문 분류 프롬프트
# --------------------------------------------------------------------------
# 입력 변수: {history}, {last_domain}, {last_subtask}, {question}
# 출력: JSON {"domain": "...", "subtask": "..."}
#
# 4개 도메인 × 12개 서브태스크의 정의/키워드를 프롬프트에 포함하여
# LLM이 정확한 분류를 할 수 있도록 유도.
# 멀티턴 규칙: 이전 분류 결과를 참조하여 연속된 문의를 동일 카테고리로 유지.
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
     - 키워드: 엔카, 무수수료, 플랫폼, 상담조회

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
2. **멀티턴 유지**: 완전히 다른 카테고리를 명시하는 질문이 아닌 이상, 이전 대화와 동일한 카테고리로 분류
   - 예: 1) "론할부 운영기준 알려줘" → 중고승용/(론/할부)  2) "nice 등급은?" → 중고승용/(론/할부) (유지)
   - 예: 1) "론할부 운영기준 알려줘" → 중고승용/(론/할부)  2) "신용구제 금리는?" → 중고승용/(신용구제) (전환)
   - 예: 1) "론할부 운영기준 알려줘" → 중고승용/(론/할부)  2) "재고금융 한도는?" → 전략금융/(재고금융) (전환)
3. 분류 불가하거나 너무 모호한 질문: domain="재질의", subtask="재질의"

### 이전 대화 맥락
- 이전 도메인: {last_domain}
- 이전 서브태스크: {last_subtask}

### 이전 대화 히스토리
{history}

### 현재 질문
{question}

### 출력 형식 (JSON만 출력, 다른 텍스트 없이)
```json
{{"domain": "중고승용|전략금융|중고리스|중형트럭|재질의", "subtask": "(론/할부)|(임직원대출)|(신용구제)|(Dual Offer)|(엔카)|(재고금융)|(제휴점 운영자금)|(매매상사 운영자금)|(운영자금 자금용도 기준)|(임차보증금)|(중고리스)|(중형트럭)|(재질의)"}}
```"""


# ==========================================================================
# REACT_STEP_PROMPT - ReAct 1스텝 판단 프롬프트
# --------------------------------------------------------------------------
# 입력 변수: {domain}, {subtask}, {available_subtasks}, {observations}, {question}
# 출력: JSON {"action": "SEARCH"/"FINISH", ...}
#
# LLM에게 이전 검색 결과를 보여주고, 추가 검색이 필요한지 판단하게 함.
# SEARCH: 새로운 query + subtask로 추가 검색 지시
# FINISH: 관련 문서 인덱스를 지정하고 답변 생성 단계로 이동
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
# GENERATE_PROMPT - 최종 답변 생성 프롬프트
# --------------------------------------------------------------------------
# 입력 변수: {domain}, {subtask}, {history}, {context}, {question}
# 출력: 자연어 한국어 답변 (마크다운 형식)
#
# RAG 패턴: 검색된 참고자료(context)를 기반으로 답변 생성.
# 답변 지침: 인사 금지, 한국어, 가독성(줄바꿈, 볼드), 간결한 답변.
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
# REQUERY_PROMPT - 재질의 안내 프롬프트
# --------------------------------------------------------------------------
# 입력 변수: {history}, {question}
# 출력: 자연어 한국어 안내 메시지 (추천 질문 포함)
#
# 분류 불가 시 사용. 지원 업무 영역 목록을 안내하고 구체적 질문을 유도.
# ==========================================================================
REQUERY_PROMPT = """너는 JB우리캐피탈 오토운영팀 챗봇이야.
지금 사용자가 업무 분류에 벗어난 질문을 한 상태야.
재질문으로 사용자에게 다시 입력을 유도해.

우리가 지원하는 업무 영역은 다음과 같아:
- **중고승용**: 론/할부, 임직원대출(ESM), 신용구제, Dual Offer, 엔카
- **전략금융**: 재고금융, 제휴점 운영자금, 매매상사 운영자금, 운영자금 자금용도 기준, 임차보증금
- **중고리스**: 중고리스
- **중형트럭**: 중형트럭

사용자 질문이나 대화내용을 참고해서 더 나은 질문에 대해서 답변해줘.
예를들어 "운영기준 알려줘"와 같은 너무 넓은 범위 질문이 들어오면,
추천질문으로 "론/할부 운영기준 알려줘", "재고금융 운영기준 알려줘" 와 같은 추천질문을 제공해줘.
너는 반드시 "한국어"로 답변해.

대화내용 : {history}

사용자 질문: {question}"""


# ==========================================================================
# FOLLOWUP_PROMPT - 후속 추천 질문 생성 프롬프트 (v3)
# --------------------------------------------------------------------------
# 입력 변수: {domain}, {subtask}, {other_subtasks}, {context}, {question}, {answer}
# 출력: JSON {"follow_ups": ["질문1", "질문2", "질문3"]}
#
# 참고자료에 포함된 정보만을 기반으로 질문 생성 (환각 방지).
# 질문 전략: 1) 같은 서브태스크 미다뤄진 내용, 2) 깊이 파기, 3) 타 서브태스크
# ==========================================================================
FOLLOWUP_PROMPT = """당신은 JB우리캐피탈 오토운영팀 챗봇의 후속 질문 생성기입니다.
사용자에게 도움이 될 만한 후속 질문 3개를 생성하세요.

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
2. 첫 번째 질문: 현재 답변과 동일한 업무({subtask}) 내에서, 참고자료에 있지만 답변에서 다루지 않은 세부 내용에 대한 질문
3. 두 번째 질문: 현재 답변의 내용을 더 깊이 파고드는 질문 (예: 조건, 예외, 구체적 수치 등)
4. 세 번째 질문: 같은 도메인의 다른 업무와 연관된 질문 (다른 업무가 없으면 현재 업무의 다른 측면)
5. 사용자 관점에서 자연스러운 한국어 질문으로 작성하세요.
6. 이미 답변된 내용을 그대로 다시 묻는 질문은 피하세요.

### 출력 형식 (JSON만 출력, 다른 텍스트 없이)
{{"follow_ups": ["질문1", "질문2", "질문3"]}}"""
