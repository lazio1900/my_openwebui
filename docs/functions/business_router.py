"""
title: 업무 라우팅 챗봇
description: 사용자 질문을 론할부/임직원대출/기타 업무로 분류하여 각 전문 챗봇으로 라우팅합니다.
author: assistant
version: 0.1.0
"""

from pydantic import BaseModel, Field
from typing import Optional, AsyncGenerator
import json


class Pipe:
    class Valves(BaseModel):
        """관리자 설정"""

        model_id: str = Field(
            default="",
            description="라우팅 및 응답 생성에 사용할 기본 모델 ID (예: ollama/llama3, gpt-4o)",
        )
        enable_auto_routing: bool = Field(
            default=True,
            description="True: LLM이 자동 분류 / False: 사용자가 직접 선택",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.BUSINESS_CATEGORIES = {
            "론할부": {
                "keywords": [
                    "론",
                    "할부",
                    "론할부",
                    "오토론",
                    "자동차할부",
                    "차량할부",
                    "할부금",
                    "할부이자",
                    "할부원금",
                    "할부기간",
                    "중도상환",
                    "만기",
                    "잔여원금",
                    "월납입",
                    "자동차론",
                    "캐피탈",
                ],
                "system_prompt": """당신은 론할부 업무 전문 상담사입니다.

## 담당 업무 범위
- 자동차 할부금융 (오토론)
- 할부 이자율 및 상환 조건 안내
- 중도상환 절차 및 수수료
- 할부 기간 변경 및 연장
- 월 납입금 조회 및 변경
- 잔여 원금 조회
- 만기 안내 및 처리

## 응답 지침
1. 정확한 금융 용어를 사용하되 고객이 이해하기 쉽게 설명하세요.
2. 금리, 수수료 등 숫자 관련 안내 시 반드시 "정확한 내용은 담당자 확인이 필요합니다"라고 안내하세요.
3. 개인정보(계좌번호, 주민번호 등)를 요청하지 마세요.
4. 론할부 업무 범위를 벗어나는 질문은 해당 업무 담당으로 안내하세요.""",
            },
            "임직원대출": {
                "keywords": [
                    "임직원",
                    "직원대출",
                    "사내대출",
                    "복리후생",
                    "임직원대출",
                    "직원론",
                    "사내론",
                    "우대금리",
                    "직원할인",
                    "재직자",
                    "퇴직금",
                    "연봉담보",
                    "급여담보",
                    "사원대출",
                ],
                "system_prompt": """당신은 임직원대출 업무 전문 상담사입니다.

## 담당 업무 범위
- 임직원 전용 대출 상품 안내
- 대출 자격 요건 확인 (재직 기간, 직급 등)
- 우대 금리 및 한도 안내
- 대출 신청 절차 안내
- 상환 방법 및 조건
- 대출 연장 및 변경
- 퇴직 시 대출 처리 절차

## 응답 지침
1. 임직원 복리후생 제도의 일환임을 안내하세요.
2. 구체적인 금리와 한도는 인사팀 또는 담당 부서 확인을 권고하세요.
3. 개인정보(사번, 주민번호 등)를 요청하지 마세요.
4. 임직원대출 범위를 벗어나는 질문은 해당 업무 담당으로 안내하세요.""",
            },
            "기타": {
                "keywords": [],
                "system_prompt": """당신은 금융 업무 일반 상담사입니다.

## 담당 업무 범위
- 론할부, 임직원대출에 해당하지 않는 일반 문의
- 업무 안내 및 담당 부서 연결
- 일반적인 금융 상담

## 응답 지침
1. 질문 내용을 파악하여 적절한 업무 담당으로 안내하세요.
2. 론할부 관련 문의는 론할부 담당으로, 임직원대출 관련 문의는 임직원대출 담당으로 안내하세요.
3. 일반적인 문의에는 친절하고 정확하게 답변하세요.
4. 개인정보를 요청하지 마세요.""",
            },
        }

    def _classify_by_keywords(self, message: str) -> str:
        """키워드 기반으로 업무를 분류합니다."""
        message_lower = message.lower()
        scores = {}
        for category, config in self.BUSINESS_CATEGORIES.items():
            if category == "기타":
                continue
            score = sum(1 for kw in config["keywords"] if kw in message_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return "기타"
        return max(scores, key=scores.get)

    def _build_classification_prompt(self, message: str) -> list:
        """LLM 기반 분류를 위한 프롬프트를 생성합니다."""
        return [
            {
                "role": "system",
                "content": """사용자의 질문을 아래 3개 업무 중 하나로 분류하세요.

1. 론할부 - 자동차 할부금융, 오토론, 할부 상환, 할부 이자 등
2. 임직원대출 - 임직원 전용 대출, 사내대출, 복리후생 대출 등
3. 기타 - 위 두 가지에 해당하지 않는 일반 문의

반드시 아래 JSON 형식으로만 응답하세요:
{"category": "론할부"} 또는 {"category": "임직원대출"} 또는 {"category": "기타"}""",
            },
            {"role": "user", "content": message},
        ]

    async def _classify_by_llm(
        self, message: str, __event_emitter__=None, generate=None
    ) -> str:
        """LLM을 사용하여 업무를 분류합니다. 실패 시 키워드 분류로 폴백합니다."""
        if not generate:
            return self._classify_by_keywords(message)

        try:
            classification_messages = self._build_classification_prompt(message)
            response = await generate(
                {
                    "model": self.valves.model_id,
                    "messages": classification_messages,
                    "stream": False,
                    "max_tokens": 50,
                }
            )

            # 응답에서 JSON 추출
            response_text = (
                response if isinstance(response, str) else str(response)
            )

            # JSON 파싱 시도
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(response_text[start:end])
                category = parsed.get("category", "기타")
                if category in self.BUSINESS_CATEGORIES:
                    return category

            return self._classify_by_keywords(message)

        except Exception:
            return self._classify_by_keywords(message)

    async def pipe(
        self,
        body: dict,
        __user__: dict = None,
        __event_emitter__=None,
        __model__: dict = None,
        **kwargs,
    ) -> dict:
        """메인 파이프: 메시지를 분류하고 해당 업무 챗봇으로 라우팅합니다."""

        messages = body.get("messages", [])
        if not messages:
            return body

        # 마지막 사용자 메시지 추출
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_message = content
                elif isinstance(content, list):
                    user_message = " ".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                break

        if not user_message:
            return body

        # 업무 분류
        generate = kwargs.get("__generate__") or kwargs.get("generate")

        if self.valves.enable_auto_routing and self.valves.model_id and generate:
            category = await self._classify_by_llm(
                user_message, __event_emitter__, generate
            )
        else:
            category = self._classify_by_keywords(user_message)

        # 분류 결과 이벤트 발행 (프론트엔드에 표시)
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"업무 분류: [{category}]",
                        "done": True,
                    },
                }
            )

        # 해당 업무의 시스템 프롬프트 적용
        system_prompt = self.BUSINESS_CATEGORIES[category]["system_prompt"]

        # 기존 시스템 메시지 교체 또는 추가
        new_messages = []
        has_system = False
        for msg in messages:
            if msg.get("role") == "system":
                has_system = True
                new_messages.append({"role": "system", "content": system_prompt})
            else:
                new_messages.append(msg)

        if not has_system:
            new_messages.insert(0, {"role": "system", "content": system_prompt})

        body["messages"] = new_messages
        return body
