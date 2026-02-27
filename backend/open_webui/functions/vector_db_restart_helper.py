"""
title: 벡터DB Pipe (자동 재시작 헬퍼 포함)
description: ChromaDB 벡터 저장 함수. pipe() 완료 후 연결된 RAG 챗봇 함수 4개의 캐시를 무효화합니다.
author: user
version: 0.4.0

사용법:
  기존 벡터DB 함수에 아래를 추가하세요:

  1. CHATBOT_FUNCTION_IDS 리스트에 챗봇 함수 ID 4개 입력
  2. pipe() 맨 끝(return 직전)에 _invalidate_chatbot_cache() 호출 추가
"""

from pydantic import BaseModel, Field


# ============================================================
# 재시작할 챗봇 함수 ID 목록 (하드코딩)
# 여기에 실제 챗봇 함수 ID 4개를 입력하세요
# ============================================================
CHATBOT_FUNCTION_IDS = [
    "chatbot_function_1",
    "chatbot_function_2",
    "chatbot_function_3",
    "chatbot_function_4",
]


class Valves(BaseModel):
    pass


class Pipe:
    def __init__(self):
        self.valves = Valves()

        # ── 기존 ChromaDB 초기화 코드는 여기에 유지 ──
        # import chromadb
        # self.client = chromadb.PersistentClient(path=...)
        # self.collection = self.client.get_or_create_collection(...)

    def _invalidate_chatbot_cache(self):
        """
        챗봇 함수 4개의 캐시를 무효화합니다.
        pipe()에서 벡터DB 처리가 완료된 후 호출하세요.
        """
        try:
            from open_webui.main import app

            for cid in CHATBOT_FUNCTION_IDS:
                if hasattr(app.state, "FUNCTIONS") and cid in app.state.FUNCTIONS:
                    del app.state.FUNCTIONS[cid]
                if hasattr(app.state, "FUNCTION_CONTENTS") and cid in app.state.FUNCTION_CONTENTS:
                    del app.state.FUNCTION_CONTENTS[cid]
                print(f"[vector_db] '{cid}' 캐시 무효화 완료")

            print(f"[vector_db] 챗봇 함수 {len(CHATBOT_FUNCTION_IDS)}개 무효화 완료")
        except Exception as e:
            print(f"[vector_db] 캐시 무효화 실패: {e}")

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__=None,
        __event_emitter__=None,
        **kwargs,
    ) -> str:
        # ── 기존 벡터DB 처리 로직 ──
        # messages = body.get("messages", [])
        # user_message = messages[-1]["content"] if messages else ""
        # self.collection.add(documents=[...], ids=[...], metadatas=[...])
        # ...

        # ── 벡터DB 처리 완료 후 챗봇 캐시 무효화 ──
        self._invalidate_chatbot_cache()

        return "벡터DB 파이프 실행 완료 (템플릿 - 실제 로직을 추가하세요)"
