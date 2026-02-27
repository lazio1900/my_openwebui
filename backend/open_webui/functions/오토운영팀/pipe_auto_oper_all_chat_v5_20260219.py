############################################
# 작성자 : 이민재
# 생성일자 : 2025-12-24
# 이력
# 2025-12-24 : 최초 생성 
# 2025-12-24 : 중고승용, 전략금융 추가
# 2026-01-14 v2 : 중고리스 추가
# 2026-01-16 v3 : 중형트럭 추가
# 2026-02-02 v4 : 프롬프트 수정
# 2026-02-19 v5 : 라우팅 기능 개선
############################################

from pydantic import BaseModel, Field
from typing import Union, Generator, Iterator, List
import re
from fastapi import Request
import json
from fastapi.responses import StreamingResponse
from open_webui.models.users import Users
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.misc import get_last_user_message


class Pipe:
    class Valves(BaseModel):
        pass

    def __init__(self):
        pass

    async def pipe(self, body: dict, __user__: dict, __request__: Request) -> str:
        try:
            # user_msg = body["messages"][-1]["content"]
            user_messages = []
            assis_messages = []
            combined_message = ""

            for item in body["messages"]:
                if item["role"] == "user":
                    user_messages.append(item["content"])
                if item["role"] == "assistant":
                    match = re.search(r"\[(.*?)\]", item["content"])
                    category_txt = match.group(1)
                    assis_messages.append(category_txt)

            if len(user_messages) > 1 and len(assis_messages) > 0:
                history_msg = user_messages[-6:-1]
                for idx, hm in enumerate(history_msg):
                    if len(hm) > 500:
                        combined_message += (
                            f"선택 모델: {assis_messages[idx]}\n"
                            + hm
                            # + hm[:200]
                            # + "\n(중간 생략)\n"
                            # + hm[-200:]
                            + "\n--------\n"
                        )
                    else:
                        combined_message += (
                            f"선택 모델: {assis_messages[idx]}\n" + hm + "\n--------\n"
                        )
            else:
                combined_message = "없음 (첫 질문)"

            user_msg = get_last_user_message(body["messages"])
            if len(user_msg) > 1000:
                user_msg = user_msg[:450] + "\n...(중간 생략)\n" + user_msg[-450:]

            question_user = PROMPT_TEMPLATE.replace("{question}", user_msg).replace(
                "{history}", combined_message
            )
            messages = body["messages"]

            data_json = {
                "model": "gpt-oss-120b",
                "messages": [{"role": "user", "content": question_user}],
                "stream": False,
                "temperature": 0,
            }
            user = Users.get_user_by_id(__user__["id"])
            response = await generate_chat_completion(__request__, data_json, user)
            res_msg = response["choices"][0]["message"]["content"]

            pattern = r"<결과>(.*?)</결과>"
            match = re.search(pattern, res_msg, re.DOTALL)
            category = match.group(1)

            model_mapping = {
                "중고승용_운영기준": "auto_oper_standard_used_car",
                "전략금융_운영기준": "auto_oper_standard_stg_fin",
                "중고리스_운영기준": "auto_oper_standard_used_lease",
                "중형트럭_운영기준": "auto_oper_standard_mid_truck",
            }

            prefix_mapping = {
                "중고승용_운영기준": "중고승용 운영기준",
                "전략금융_운영기준": "전략금융 운영기준",
                "중고리스_운영기준": "중고리스 운영기준",
                "중형트럭_운영기준": "중형트럭 운영기준",
            }

            model = model_mapping.get(category)
            prefix_str = prefix_mapping.get(category)

            if model in ["gpt-oss-120b"]:
                data_json = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                }
            else:
                data_json = {
                    "model": model,
                    "messages": [{"role": "user", "content": user_msg}],
                    "stream": True,
                }
            # 모델명 맨앞에 추가하지않고 그냥 응답
            # return await generate_chat_completion(__request__, data_json, user)
            original_response = await generate_chat_completion(
                __request__, data_json, user
            )

            async def stream_with_prefix():
                prefix_sent = False
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
                                if content and not prefix_sent:
                                    choice["delta"]["content"] = (
                                        f"#### [{prefix_str}]\n" + content
                                    )
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

                suffix_str = (
                    f"{prefix_str}입니다. 다른 운영기준이 필요한 경우 '새 채팅'을 이용해주세요."
                )
                suffix_data = {
                    "choices": [{"delta": {"content": f"\n\n **[{suffix_str}]**"}}]
                }
                suffix_chunk = f"data: {json.dumps(suffix_data, ensure_ascii=False)}"
                yield suffix_chunk.encode("utf-8")

            return StreamingResponse(
                stream_with_prefix(), media_type="text/plain; charset=utf-8"
            )

        except Exception as e:
            # data_json = {
            #     "model": "gemma3:27b",
            #     "messages": messages,
            #     "stream": True,
            # }
            # # 모델명 맨앞에 추가하지않고 그냥 응답
            # return await generate_chat_completion(__request__, data_json, user)

            error_message = res_msg + str(data_json) + str(e)
            error_data = {
                "choices": [
                    {"delta": {"content": f"\n\n#### [Error]\n{error_message}\n"}}
                ],
            }
            error_chunk = f"data: {json.dumps(error_data, ensure_ascii=False)}"

            async def stream_error():
                yield error_chunk.encode("utf-8")

            return StreamingResponse(
                stream_error(), media_type="text/plain; charset=utf-8"
            )




PROMPT_TEMPLATE = """
당신은 아래 4개의 카테고리 중 하나에만 **정확히** 질문을 매핑해야 합니다.
다른 텍스트(설명, 이유, 부연)는 절대 출력하지 말고, 반드시 지정된 형식만 사용합니다.
<결과>카테고리명</결과>

### 카테고리와 핵심 키워드
1. 전략금융_운영기준
   카테고리 : '재고금융', '제휴점 운영자금', '매매상사 운영자금', '운영자금 자금용도 기준', '임차보증금'
   키워드 : 재고금융, 제휴점, 매매상사, 한도, 금리, LTV, 서류, 연체, 사후관리, Penalty
2. 중고승용_운영기준
   카테고리 : '론/할부', '임직원대출(ESM)', '신용구제', 'Dual Offer', '엔카 무수수료'
   키워드 : Dual_C, Dual_O, 채널, 신용구제, 신용등급, 금리, 대출개월수, 중도상환수수료, 판촉수수료, 엔카, 중고승용
3. 중고리스_운영기준
   카테고리 : '중고리스'
   키워드 : 운용리스, 금융리스, 잔가, 잔가율, 그룹 I-P, 수입차, 서류, 보험, 세금
4. 중형트럭_운영기준
   카테고리 : '중형트럭'
   키워드 : 중형트럭, 할부, 대출, 금리, LTV, 서류, 특장, 주행거리, 예외협의

### 사용자 멀티턴 대화 유의
- 사용자는 한번 분류된 업무에서 연속적으로 질문 할 수 있습니다.
- 완전히 다른 카테고리 질문이 아닌이상 동일한 카테고리로 진행하세요
- 예시. 1) 론할부 운영기준 알려줘 : 중고승용_운영기준 > 2) nice 등급은 어떻게 되는데 ? : 중고승용_운영기준

### Few-shot 예시 (각 카테고리 2개)
[예시]  
Q: “매매상사 운영자금 한도는 최근 3개월 판매대수와 NICE CB 점수에 따라 어떻게 산정됩니까?”  
A: <결과>전략금융_운영기준</결과>

Q: “Dual_O 상품을 제휴점에서 2천만원 대출받을 경우 금리는 얼마인가요?”  
A: <결과>중고승용_운영기준</결과>

Q: “수입차 리스에서 'I-P군'에 해당하는지 확인하려면 어떤 자료를 보나요?”  
A: <결과>중고리스_운영기준</결과>

Q: “9톤 특장 트럭이 구조변경 차량이면 어떤 서류가 필요합니까?”  
A: <결과>중형트럭_운영기준</결과>

---  
**입력**: 사용자가 질문을 입력하면 위 규칙·키워드·예시를 참고해 가장 적합한 카테고리를 <결과> 태그 안에 넣어 출력하십시오.

## 이전 질문 이력:
{history}

## 현재 질문:
{question}
"""