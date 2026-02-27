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
############################################

from pydantic import BaseModel
from typing import Dict, Callable, Awaitable, Optional, Protocol
import re
import time
import json
import asyncio
import sys

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI

from open_webui.utils.misc import get_last_user_message
from fastapi import Request


###################################################################################
### 비동기 이벤트 통신 유틸리티
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


###################################################################################
### 도메인/서브태스크 매핑 (한 곳에서 관리)

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

# subtask → domain 역매핑 (검증용)
SUBTASK_TO_DOMAIN = {}
for domain, info in DOMAIN_SUBTASK_MAP.items():
    for st in info["subtasks"]:
        SUBTASK_TO_DOMAIN[st] = domain


###################################################################################
### Pipe 클래스


class Pipe:
    class Valves(BaseModel):
        BASE_IMG_URL: str = "https://ai.wooricap.com/static/auto_oper_images/"
        CHROMA_PORT: int = 8800
        CHROMA_IP: str = "localhost"
        STANDARD_COLLECTION_NAME_IMAGE: str = "auto_oper_standard_image"
        STANDARD_COLLECTION_NAME_TEXT: str = "auto_oper_standard_text"
        EMBEDDING_K: int = 5
        # OpenAI API 설정 (개발환경)
        OPENAI_API_KEY: str = ""
        OPENAI_MODEL: str = "gpt-4o-mini"
        OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
        # ReAct 에이전트 설정
        REACT_MAX_ITERATIONS: int = 3

    def __init__(self):
        self.valves = self.Valves()
        self._state_by_user: Dict[str, dict] = {}
        self._state_lock = asyncio.Lock()
        self._state_ttl_sec = 60 * 60  # 1시간

        # 클라이언트는 pipe() 첫 호출 시 lazy 초기화
        self.openai_client = None
        self.chroma_db_text = None
        self.chroma_db_image = None
        self._initialized = False

        print("Orchestrator + ReAct 에이전트 Pipe 등록 완료! (lazy init)")

    def _ensure_initialized(self):
        """Valves 값이 확정된 후 클라이언트를 초기화"""
        if self._initialized:
            return

        # OpenAI 클라이언트
        self.openai_client = AsyncOpenAI(api_key=self.valves.OPENAI_API_KEY)

        # ChromaDB 클라이언트
        chroma_client = chromadb.HttpClient(
            host=self.valves.CHROMA_IP, port=self.valves.CHROMA_PORT
        )

        # 임베딩 함수 (OpenAI)
        embedding_func = OpenAIEmbeddings(
            model=self.valves.OPENAI_EMBEDDING_MODEL,
            openai_api_key=self.valves.OPENAI_API_KEY,
        )

        # Chroma 객체 2개만 생성 (text + image)
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
    # Pipe 엔트리포인트
    # =================================================================
    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] | None = None,
        __user__: dict = None,
        __request__: Request = None,
    ):
        try:
            self._ensure_initialized()
            start_time = time.time()
            user_id = self._get_user_id(body, __user__)
            user_message = get_last_user_message(body["messages"])
            messages = body["messages"]
            send_status = get_send_status(__event_emitter__)

            # TTL 지난 사용자 상태 정리 + last_classified 로드
            now = time.time()
            async with self._state_lock:
                for k, v in list(self._state_by_user.items()):
                    if now - v.get("ts", 0) > self._state_ttl_sec:
                        del self._state_by_user[k]
                user_state = self._state_by_user.get(user_id, {})
                last_classified = user_state.get("last_classified", "")
                last_domain = user_state.get("last_domain", "")

            # 상태 초기화
            state = {
                "user_message": user_message,
                "messages": messages,
                "user_id": user_id,
                "event_emitter": __event_emitter__,
                "send_status": send_status,
                "start_time": start_time,
                # 분류 결과
                "domain": "",
                "subtask": "",
                "is_requery": False,
                # 검색 결과
                "filtered_docs": [],
                # 멀티턴 상태
                "is_first_entry": False,
                "last_classified": last_classified,
                "last_domain": last_domain,
            }

            # ── 1. Orchestrator: 분류 ──
            state = await self._classify(state)

            # ── 2. 재질의 분기 ──
            if state["is_requery"]:
                return await self._requery(state)

            # ── 3. ReAct 에이전트: 검색 + 문서 수집 ──
            state = await self._react_agent(state)

            # ── 4. 최종 답변 생성 (스트리밍) ──
            return await self._generate(state)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"exception :: {e}\n라인: {exc_tb.tb_lineno}"
            send_status = get_send_status(__event_emitter__)
            await send_status(status_message=f"오류 발생: {e}", done=True)
            return f"\n\n#### [Error]\n{error_message}\n"

    # =================================================================
    # Orchestrator: 분류
    # =================================================================
    async def _classify(self, state: dict) -> dict:
        send_status = state["send_status"]
        await send_status(status_message="질문 분류 중 ...", done=False)

        # 히스토리 요약 (최근 5턴)
        history_summary = self._build_history_summary(state["messages"])

        # 통합 분류 프롬프트
        prompt = CLASSIFY_PROMPT.format(
            history=history_summary,
            last_domain=state["last_domain"] or "없음",
            last_subtask=state["last_classified"] or "없음",
            question=state["user_message"],
        )

        content = await self._llm_call(prompt)

        # JSON 파싱
        domain, subtask = self._parse_classification(content)

        # 첫 진입 여부 판단
        is_first_entry = (
            state["last_classified"] == ""
            or state["last_classified"] != subtask
        )

        state["domain"] = domain
        state["subtask"] = subtask
        state["is_requery"] = domain == "재질의"
        state["is_first_entry"] = is_first_entry

        if not state["is_requery"]:
            display = DOMAIN_SUBTASK_MAP.get(domain, {}).get(
                "display_name", domain
            )
            await send_status(
                status_message=f"분류 결과 : {display} > {subtask}", done=True
            )
            # 상태 저장
            await self._update_state(state["user_id"], domain, subtask)

        return state

    # =================================================================
    # ReAct 에이전트
    # =================================================================
    async def _react_agent(self, state: dict) -> dict:
        """ReAct 루프: Think→Act→Observe 반복으로 관련 문서 수집"""
        observations = []  # [{"query": str, "subtask": str, "docs": list, "doc_summaries": list}, ...]
        send_status = state["send_status"]

        for i in range(self.valves.REACT_MAX_ITERATIONS):
            await send_status(
                status_message=f"ReAct 에이전트 실행 중... (반복 {i + 1}/{self.valves.REACT_MAX_ITERATIONS})",
                done=False,
            )

            # ReAct Step: LLM이 다음 행동 결정
            decision = await self._react_step(state, observations)

            if decision["action"] == "FINISH":
                # 에이전트가 충분한 정보를 수집했다고 판단
                relevant_indices = decision.get("relevant_doc_indices", [])
                all_docs = []
                for obs in observations:
                    all_docs.extend(obs["docs"])

                # 관련 문서만 필터링 (에이전트가 선택)
                if relevant_indices and all_docs:
                    state["filtered_docs"] = [
                        all_docs[idx] for idx in relevant_indices
                        if 0 <= idx < len(all_docs)
                    ]
                else:
                    state["filtered_docs"] = all_docs

                # 필터링 결과가 0개면 안전장치로 전체 사용
                if not state["filtered_docs"] and all_docs:
                    state["filtered_docs"] = all_docs
                break

            elif decision["action"] == "SEARCH":
                # 도구 실행: ChromaDB 벡터 검색
                query = decision["query"]
                subtask = decision["subtask"]

                await send_status(
                    status_message=f"{subtask} 검색 중: \"{query[:30]}...\"" if len(query) > 30 else f"{subtask} 검색 중: \"{query}\"",
                    done=False,
                )

                docs = self._execute_tool(query, subtask)

                # 전체 문서 인덱스 오프셋 계산 (이전 관측에 쌓인 문서 수)
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
            # 최대 반복 도달 → 수집된 모든 문서 사용
            all_docs = []
            for obs in observations:
                all_docs.extend(obs["docs"])
            state["filtered_docs"] = all_docs

        await send_status(
            status_message=f"관련 문서 {len(state['filtered_docs'])}건 수집 완료",
            done=True,
        )
        return state

    async def _react_step(self, state: dict, observations: list) -> dict:
        """1회 LLM 호출: 다음 행동 결정 (SEARCH or FINISH)"""
        domain = state["domain"]
        subtask = state["subtask"]
        domain_info = DOMAIN_SUBTASK_MAP.get(domain, {})
        available_subtasks = ", ".join(domain_info.get("subtasks", [subtask]))

        # 이전 관측 결과 텍스트 생성
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
        return self._parse_react_decision(content, state)

    def _execute_tool(self, query: str, subtask: str) -> list:
        """ChromaDB 벡터 검색 (LLM 호출 없음)"""
        return self.chroma_db_text.similarity_search(
            query=query,
            k=self.valves.EMBEDDING_K,
            filter={"subtask": subtask},
        )

    # =================================================================
    # 답변 생성 (스트리밍)
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

        # 첫 진입시: 이미지 + non-stream 답변
        if state["is_first_entry"] and DOMAIN_SUBTASK_MAP.get(domain, {}).get("has_image", False):
            return await self._generate_with_image(state)

        # 답변 생성 프롬프트
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

        end_time = time.time()
        exe_time = end_time - state["start_time"]
        await send_status(status_message=f"완료: {exe_time:.2f}초", done=True)

        # 출처 표시
        send_citation = get_send_citation(state["event_emitter"])
        for idx, doc in enumerate(filtered_docs, start=1):
            await send_citation(
                url=f"출처{idx}",
                title=f"출처{idx}",
                content=doc.page_content,
            )

        # OpenAI 스트리밍 → AsyncGenerator로 반환
        prefix = f"#### [{display_name} > {subtask}]\n"
        suffix = f"\n\n **[{display_name}입니다. 다른 운영기준이 필요한 경우 '새 채팅'을 이용해주세요.]**"

        async def stream_response():
            yield prefix
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
            yield suffix

        return stream_response()

    async def _generate_with_image(self, state: dict):
        """첫 진입시 이미지 + 답변 생성"""
        domain = state["domain"]
        subtask = state["subtask"]
        filtered_docs = state["filtered_docs"]
        user_message = state["user_message"]
        messages = state["messages"]
        display_name = DOMAIN_SUBTASK_MAP.get(domain, {}).get(
            "display_name", domain
        )

        # 이미지 검색
        image_results = self.chroma_db_image.similarity_search(
            query=subtask, k=1
        )

        image_url = ""
        if image_results:
            image_url = image_results[0].metadata.get("html_images", "")
        if not image_url:
            image_url = "이미지 검색 결과 없음"

        # non-stream으로 답변 생성
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

        llm_text = await self._llm_call(prompt)

        # 이미지 + 답변 결합
        combined = f"#### [{display_name} > {subtask}]\n{image_url}\n{llm_text}"

        def stream_output():
            for char in combined:
                yield char

        return stream_output()

    # =================================================================
    # 재질의 (스트리밍)
    # =================================================================
    async def _requery(self, state: dict):
        send_status = state["send_status"]
        await send_status(status_message="재질의 응답 생성 중...", done=False)

        prompt = REQUERY_PROMPT.format(
            history=str(state["messages"][-6:]),
            question=state["user_message"],
        )

        end_time = time.time()
        exe_time = end_time - state["start_time"]
        await send_status(status_message=f"완료: {exe_time:.2f}초", done=True)

        # OpenAI 스트리밍 → AsyncGenerator
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
    async def _llm_call(self, prompt: str) -> str:
        """non-stream LLM 호출 (단일 프롬프트) → 텍스트 반환"""
        response = await self.openai_client.chat.completions.create(
            model=self.valves.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=0,
        )
        return response.choices[0].message.content

    async def _llm_call_messages(self, messages: list) -> str:
        """non-stream LLM 호출 (messages 배열) → 텍스트 반환"""
        response = await self.openai_client.chat.completions.create(
            model=self.valves.OPENAI_MODEL,
            messages=messages,
            stream=False,
            temperature=0,
        )
        return response.choices[0].message.content

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
        """최근 대화 히스토리를 요약"""
        if len(messages) <= 1:
            return "없음 (첫 질문)"

        history_parts = []
        recent = messages[-7:-1]  # 최근 6개 메시지 (마지막 제외)
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if len(content) > 300:
                content = content[:150] + "\n...(중략)...\n" + content[-150:]
            history_parts.append(f"[{role}] {content}")

        return "\n".join(history_parts) if history_parts else "없음 (첫 질문)"

    def _parse_classification(self, content: str) -> tuple:
        """분류 LLM 응답에서 domain, subtask 파싱"""
        try:
            # JSON 파싱 시도
            cleaned = re.sub(r"```(?:json)?", "", content).strip().strip("`")
            data = json.loads(cleaned)
            domain = data.get("domain", "재질의")
            subtask = data.get("subtask", "재질의")

            # 유효성 검증
            if domain in DOMAIN_SUBTASK_MAP:
                valid_subtasks = DOMAIN_SUBTASK_MAP[domain]["subtasks"]
                if subtask not in valid_subtasks:
                    # subtask가 정확하지 않으면 가장 유사한 것 찾기
                    for vs in valid_subtasks:
                        if subtask.replace("(", "").replace(")", "") in vs:
                            subtask = vs
                            break
                    else:
                        # 도메인에 서브태스크가 1개이면 자동 매핑
                        if len(valid_subtasks) == 1:
                            subtask = valid_subtasks[0]
                return domain, subtask
            else:
                return "재질의", "재질의"
        except (json.JSONDecodeError, KeyError, AttributeError):
            # JSON 파싱 실패 → 텍스트에서 추출 시도
            for domain, info in DOMAIN_SUBTASK_MAP.items():
                for st in info["subtasks"]:
                    if st in content:
                        return domain, st
            return "재질의", "재질의"

    def _parse_react_decision(self, content: str, state: dict) -> dict:
        """ReAct Step LLM 응답에서 action 파싱"""
        try:
            cleaned = re.sub(r"```(?:json)?", "", content).strip().strip("`")
            data = json.loads(cleaned)
            action = data.get("action", "FINISH")

            if action == "SEARCH":
                query = data.get("query", state["user_message"])
                subtask = data.get("subtask", state["subtask"])

                # subtask 유효성 검증
                domain_info = DOMAIN_SUBTASK_MAP.get(state["domain"], {})
                valid_subtasks = domain_info.get("subtasks", [])
                if subtask not in valid_subtasks:
                    # 유사한 것 찾기
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
                # 알 수 없는 action → FINISH로 처리
                return {"action": "FINISH", "relevant_doc_indices": []}

        except (json.JSONDecodeError, KeyError, AttributeError):
            # JSON 파싱 실패 → 기본 검색 수행
            return {
                "action": "SEARCH",
                "query": state["user_message"],
                "subtask": state["subtask"],
            }


###################################################################################
### 프롬프트 템플릿
###################################################################################

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
