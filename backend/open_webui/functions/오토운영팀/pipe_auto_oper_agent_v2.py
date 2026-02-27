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
# 2026-02-27 v2(운영) : 운영환경 적용
#   - generate_chat_completion()으로 LLM 호출 전환 (OpenAI 직접 호출 제거)
#   - SentenceTransformerEmbeddings(kure)로 임베딩 전환
#   - StreamingResponse 인터셉트 방식으로 스트리밍 답변 생성
# 2026-02-27 v3(운영) : 재질의 고도화 + 추천질문 개선
#   - 재질의 사유 세분화 (업무외/모호/정보부족) → 상황별 맞춤 안내
#   - CLASSIFY_PROMPT에 requery_reason, requery_detail 출력 추가
#   - REQUERY_PROMPT: 사유별 응답 전략 + 바로 입력 가능한 추천 질문
#   - FOLLOWUP_PROMPT: 업무 처리 유도형 추천 질문으로 개선
#   - _generate_with_image()에 citation 전송 추가
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
from langchain_community.embeddings import SentenceTransformerEmbeddings  # 로컬 임베딩 (kure)

from open_webui.utils.misc import get_last_user_message  # 대화에서 마지막 사용자 메시지 추출
from open_webui.models.users import Users                # 사용자 DB 조회
from open_webui.utils.chat import generate_chat_completion  # Open WebUI 내장 LLM 호출
from fastapi import Request             # Open WebUI 요청 객체
from fastapi.responses import StreamingResponse  # SSE 스트리밍 응답


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
        EMBED_PATH: str = "/data1/embedding/kure"    # SentenceTransformer 임베딩 모델 경로
        CHROMA_PORT: int = 8800                       # ChromaDB 포트
        CHROMA_IP: str = "172.18.237.81"             # ChromaDB 호스트 (운영 서버)
        LLM_MODEL_NAME: str = "gpt-oss-120b"         # LLM 모델명 (Ollama)
        BASE_IMG_URL: str = "https://ai.wooricap.com/static/auto_oper_images/"
        STANDARD_COLLECTION_NAME_IMAGE: str = "auto_oper_standard_image"
        STANDARD_COLLECTION_NAME_TEXT: str = "auto_oper_standard_text"
        EMBEDDING_K: int = 5                          # 벡터 검색 시 반환할 문서 수
        # ReAct 에이전트 설정
        REACT_MAX_ITERATIONS: int = 3                 # Think→Act→Observe 최대 반복 횟수
        # 후속 질문 생성 설정
        ENABLE_FOLLOWUPS: bool = True                 # False로 하면 추천 질문 미표시

    def __init__(self):
        self.valves = self.Valves()
        self._state_by_user: Dict[str, dict] = {}  # 사용자별 멀티턴 상태 저장소
        self._state_lock = asyncio.Lock()           # 동시 접근 방지 Lock
        self._state_ttl_sec = 60 * 60               # 1시간 후 상태 자동 삭제

        # ★ Eager 초기화: 운영환경에서는 Valves 기본값이 운영 설정이므로
        #   __init__ 시점에서 바로 임베딩 + ChromaDB 초기화
        chroma_client = chromadb.HttpClient(
            host=self.valves.CHROMA_IP, port=self.valves.CHROMA_PORT
        )

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

        print("Orchestrator + ReAct 에이전트 Pipe 등록 완료! (v3 운영환경)")

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
    #   - StreamingResponse (SSE 스트리밍) 또는 Generator (이미지 포함 시)
    #   - Open WebUI가 응답을 소비하여 프론트엔드에 실시간 전송
    # =================================================================
    async def pipe(
        self,
        body: dict,                        # Open WebUI가 전달하는 요청 body (messages, user 등)
        __event_emitter__: Callable[[dict], Awaitable[None]] | None = None,  # 프론트엔드 이벤트 콜백
        __user__: dict = None,             # 현재 로그인 사용자 정보
        __request__: Request = None,       # FastAPI Request 객체
    ):
        try:
            start_time = time.time()
            user_id = self._get_user_id(body, __user__)
            user_message = get_last_user_message(body["messages"])  # 마지막 사용자 메시지만 추출
            messages = body["messages"]       # 전체 대화 히스토리 (system, user, assistant 메시지)
            send_status = get_send_status(__event_emitter__)  # 상태 표시 유틸리티

            # ── User 객체 resolve (generate_chat_completion에 필요) ──
            user = Users.get_user_by_id(__user__["id"])

            # ── 새 채팅 감지 ──
            # 대화에서 user 메시지가 1개뿐이면 새 채팅으로 판단.
            # 새 채팅 시 이전 분류 상태를 초기화하여 첫 진입(이미지 포함)으로 처리.
            user_msg_count = sum(1 for m in messages if m.get("role") == "user")
            is_new_chat = user_msg_count <= 1

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

                # 새 채팅이면 이전 상태 초기화
                if is_new_chat and user_id in self._state_by_user:
                    del self._state_by_user[user_id]

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
                # generate_chat_completion에 필요한 객체
                "__request__": __request__,       # FastAPI Request 객체
                "user": user,                     # Users 모델 객체
                # _classify()가 채우는 분류 결과
                "domain": "",                     # 도메인명 (예: "중고승용")
                "subtask": "",                    # 서브태스크명 (예: "(론/할부)")
                "is_requery": False,              # 재질의 여부 (분류 불가 시 True)
                "requery_reason": "",             # 재질의 사유 (업무외/모호/정보부족)
                "requery_detail": "",             # 재질의 상세 설명
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
        prompt = CLASSIFY_PROMPT.format(
            history=history_summary,
            last_domain=state["last_domain"] or "없음",
            last_subtask=state["last_classified"] or "없음",
            question=state["user_message"],
        )

        # LLM 호출 → JSON 응답 수신
        content = await self._llm_call(prompt, state)

        # JSON에서 domain, subtask, requery_reason, requery_detail 추출 + 유효성 검증
        domain, subtask, requery_reason, requery_detail = self._parse_classification(content)

        # 첫 진입 여부 판단
        is_first_entry = (
            state["last_classified"] == ""
            or state["last_classified"] != subtask
        )

        # state에 분류 결과 기록
        state["domain"] = domain
        state["subtask"] = subtask
        state["is_requery"] = domain == "재질의"
        state["is_first_entry"] = is_first_entry
        state["requery_reason"] = requery_reason      # 업무외 / 모호 / 정보부족
        state["requery_detail"] = requery_detail      # LLM이 작성한 재질의 사유

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
    # _react_agent() - ReAct 패턴 루프 컨트롤러
    # ---------------------------------------------------------------
    # ReAct(Reasoning + Acting) 패턴: LLM이 스스로 "생각(Think)"하고,
    # "행동(Act=벡터검색)"을 수행하고, "관찰(Observe=검색결과)"한 뒤,
    # 충분한 정보가 모이면 "완료(FINISH)"를 선언하는 자율 루프.
    #
    # 최대 반복: REACT_MAX_ITERATIONS (기본 3회)
    # state 업데이트: filtered_docs
    # =================================================================
    async def _react_agent(self, state: dict) -> dict:
        """ReAct 루프: Think→Act→Observe 반복으로 관련 문서 수집"""
        observations = []
        send_status = state["send_status"]

        for i in range(self.valves.REACT_MAX_ITERATIONS):
            await send_status(
                status_message=f"ReAct 에이전트 실행 중... (반복 {i + 1}/{self.valves.REACT_MAX_ITERATIONS})",
                done=False,
            )

            # ── Think: LLM이 다음 행동을 결정 ──
            decision = await self._react_step(state, observations)

            if decision["action"] == "FINISH":
                # ── FINISH: 에이전트가 충분한 정보를 수집했다고 판단 ──
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

                # 안전장치: 필터링 후 0건이면 전체 문서로 폴백
                if not state["filtered_docs"] and all_docs:
                    state["filtered_docs"] = all_docs
                break

            elif decision["action"] == "SEARCH":
                # ── Act: ChromaDB 벡터 검색 실행 ──
                query = decision["query"]
                subtask = decision["subtask"]

                await send_status(
                    status_message=f"{subtask} 검색 중: \"{query[:30]}...\"" if len(query) > 30 else f"{subtask} 검색 중: \"{query}\"",
                    done=False,
                )

                docs = self._execute_tool(query, subtask)

                # ── Observe: 검색 결과를 관측 기록에 추가 ──
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
    # _execute_tool() - 벡터 검색 도구 실행
    # ---------------------------------------------------------------
    # ChromaDB에서 similarity_search 수행 (코사인 유사도).
    # metadata의 subtask 필드로 필터링하여 해당 서브태스크 문서만 검색.
    # EMBEDDING_K(기본 5)개의 가장 유사한 문서를 반환.
    # =================================================================
    def _execute_tool(self, query: str, subtask: str) -> list:
        """ChromaDB 벡터 검색 (LLM 호출 없음, 순수 DB 쿼리)"""
        return self.chroma_db_text.similarity_search(
            query=query,
            k=self.valves.EMBEDDING_K,
            filter={"subtask": subtask},
        )

    # =================================================================
    # _generate() - 최종 답변 생성 (StreamingResponse 인터셉트)
    # ---------------------------------------------------------------
    # ReAct 에이전트가 수집한 문서(filtered_docs)를 참고자료로 활용하여
    # 사용자 질문에 대한 최종 답변을 generate_chat_completion(stream=True)으로 생성.
    # StreamingResponse의 body_iterator를 인터셉트하여 prefix/suffix/후속질문 주입.
    #
    # 출력 구조 (SSE 청크 순서):
    #   1. prefix: "#### [중고승용 운영기준 > (론/할부)]"
    #   2. LLM 스트리밍 답변 (토큰 단위로 실시간 전송)
    #   3. suffix: "**[중고승용 운영기준입니다. 다른 운영기준이 필요한 경우...]**"
    #   4. 후속 추천 질문 (ENABLE_FOLLOWUPS=True일 때)
    #
    # 반환값: StreamingResponse (SSE)
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
        if state["is_first_entry"] and DOMAIN_SUBTASK_MAP.get(domain, {}).get("has_image", False):
            return await self._generate_with_image(state)

        # ── 참고자료 컨텍스트 구성 ──
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

        # 실행 시간 표시
        end_time = time.time()
        exe_time = end_time - state["start_time"]
        await send_status(status_message=f"완료: {exe_time:.2f}초", done=True)

        # ── 출처(citation) 전송 ──
        send_citation = get_send_citation(state["event_emitter"])
        for idx, doc in enumerate(filtered_docs, start=1):
            await send_citation(
                url=f"출처{idx}",
                title=f"출처{idx}",
                content=doc.page_content,
            )

        # ── StreamingResponse 인터셉트: prefix/suffix/후속질문 주입 ──
        prefix = f"#### [{display_name} > {subtask}]\n"
        suffix = f"\n\n **[{display_name}입니다. 다른 운영기준이 필요한 경우 '새 채팅'을 이용해주세요.]**"

        data_json = {
            "model": self.valves.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0,
        }
        original_response = await generate_chat_completion(
            state["__request__"], data_json, state["user"]
        )

        # state를 클로저 캡처하여 후속질문 생성에 사용
        _state = state
        _self = self

        async def stream_with_extras():
            prefix_sent = False
            collected_answer = []
            async for chunk in original_response.body_iterator:
                decoded_chunk = (
                    chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                )
                if decoded_chunk.startswith("data:"):
                    json_data = decoded_chunk.lstrip("data:").strip()
                    try:
                        data = json.loads(json_data)
                        choices = data.get("choices", [])
                        for choice in choices:
                            content = choice.get("delta", {}).get("content")
                            if content is not None:
                                # 답변 텍스트 캡처 (후속질문 생성용)
                                collected_answer.append(content)
                                if not prefix_sent:
                                    # 첫 번째 content 청크에 prefix 주입
                                    choice["delta"]["content"] = prefix + content
                                    prefix_sent = True
                                    break

                        modified_chunk = (
                            f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                        )
                        yield modified_chunk.encode("utf-8")
                    except json.JSONDecodeError:
                        yield chunk
                else:
                    yield chunk

            # suffix 청크 yield
            suffix_data = {
                "choices": [{"delta": {"content": suffix}}]
            }
            suffix_chunk = f"data: {json.dumps(suffix_data, ensure_ascii=False)}\n\n"
            yield suffix_chunk.encode("utf-8")

            # 후속 추천 질문 생성 (v3 기능)
            if _self.valves.ENABLE_FOLLOWUPS:
                full_answer = "".join(collected_answer)
                follow_ups = await _self._generate_followups(_state, full_answer)
                if follow_ups:
                    followup_text = "\n\n---\n**추천 질문:**\n"
                    for idx, q in enumerate(follow_ups, 1):
                        followup_text += f"{idx}. {q}\n"
                    followup_data = {
                        "choices": [{"delta": {"content": followup_text}}]
                    }
                    followup_chunk = f"data: {json.dumps(followup_data, ensure_ascii=False)}\n\n"
                    yield followup_chunk.encode("utf-8")

        return StreamingResponse(
            stream_with_extras(), media_type="text/plain; charset=utf-8"
        )

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

        # ── 출처(citation) 전송 ──
        # 이미지 경로에서도 참고자료 출처를 프론트엔드에 전달
        send_citation = get_send_citation(state["event_emitter"])
        for idx, doc in enumerate(filtered_docs, start=1):
            await send_citation(
                url=f"출처{idx}",
                title=f"출처{idx}",
                content=doc.page_content,
            )

        # non-streaming LLM 호출 (이미지와 조합해야 하므로 전체 텍스트 한번에 생성)
        llm_text = await self._llm_call(prompt, state)

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
    # _requery() - 재질의 응답 생성 (StreamingResponse 직접 반환)
    # ---------------------------------------------------------------
    # _classify()에서 domain="재질의"로 판단된 경우 호출.
    # RAG 검색 없이 LLM만으로 안내 메시지를 생성하여 사용자에게 반환.
    # generate_chat_completion(stream=True) 반환값을 그대로 반환.
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
        return response  # StreamingResponse 그대로 반환

    # =================================================================
    # 유틸리티 함수
    # =================================================================

    # -----------------------------------------------------------------
    # _llm_call() - 단일 프롬프트 LLM 호출 (non-streaming)
    # generate_chat_completion을 사용하여 LLM 호출.
    # 분류, ReAct, 후속질문 등 JSON 응답이 필요한 곳에서 사용.
    # temperature=0으로 결정론적 출력 보장.
    # -----------------------------------------------------------------
    async def _llm_call(self, prompt: str, state: dict) -> str:
        """non-stream LLM 호출 (단일 프롬프트) → 텍스트 반환"""
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
                "last_classified": subtask,
                "last_domain": domain,
                "ts": time.time(),
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
        recent = messages[-7:-1]
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if len(content) > 300:
                content = content[:150] + "\n...(중략)...\n" + content[-150:]
            history_parts.append(f"[{role}] {content}")

        return "\n".join(history_parts) if history_parts else "없음 (첫 질문)"

    # -----------------------------------------------------------------
    # _parse_classification() - 분류 LLM 응답 JSON 파싱
    # -----------------------------------------------------------------
    def _parse_classification(self, content: str) -> tuple:
        """분류 LLM 응답에서 domain, subtask, requery_reason, requery_detail 파싱 + 유효성 검증"""
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

    # -----------------------------------------------------------------
    # _parse_react_decision() - ReAct Step LLM 응답 JSON 파싱
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
# REACT_STEP_PROMPT - ReAct 1스텝 판단 프롬프트
# --------------------------------------------------------------------------
# 입력 변수: {domain}, {subtask}, {available_subtasks}, {observations}, {question}
# 출력: JSON {"action": "SEARCH"/"FINISH", ...}
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
# 입력 변수: {history}, {question}, {requery_reason}, {requery_detail}
# 출력: 자연어 한국어 안내 메시지 (상황별 맞춤 안내 + 추천 질문)
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
# FOLLOWUP_PROMPT - 후속 추천 질문 생성 프롬프트 (v3)
# --------------------------------------------------------------------------
# 입력 변수: {domain}, {subtask}, {other_subtasks}, {context}, {question}, {answer}
# 출력: JSON {"follow_ups": ["질문1", "질문2", "질문3"]}
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
