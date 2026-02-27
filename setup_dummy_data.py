"""
더미 데이터 생성 스크립트
- Outline wiki에 더미 문서 생성 (선택)
- ChromaDB에 OpenAI 임베딩으로 직접 삽입

사용법:
  python setup_dummy_data.py                          # ChromaDB만 (Outline 건너뜀)
  python setup_dummy_data.py --outline-url http://localhost:3000 --outline-api-key ol_api_xxx
"""

import argparse
import re
import sys

import numpy as np
import chromadb
from openai import OpenAI

# ============================================================
# 설정
# ============================================================
OPENAI_API_KEY = ""  # 실행 시 --openai-api-key 로 전달
CHROMA_HOST = "localhost"
CHROMA_PORT = 8800
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

TEXT_COLLECTION = "auto_oper_standard_text"
IMAGE_COLLECTION = "auto_oper_standard_image"

# ============================================================
# 더미 데이터 정의 (12개 서브태스크)
# ============================================================
# 형식: 기존 파이프라인의 text_doc_preprocesser와 동일
#   - "###" 구분자로 분리
#   - 첫 번째 괄호 "(서브태스크명)"이 metadata.subtask가 됨

DUMMY_TEXT_DOCS = {
    "중고승용 운영기준 #text": """
### (론/할부) 론/할부 운영기준
**상품명**: 중고승용차 구입대출(론/할부)
**영업채널**: 제휴점, Direct
**대출기간**: 최소 12개월 ~ 최대 84개월
**대상고객**: 내국인 개인, 법인 (외국인은 별도 심사)
**금리등급**: 1~15등급 (NICE 신용평점 기준)
- G/L금리: 등급별 기본금리 적용
- NEGO조정금리: 거점장 승인 하에 최대 -1.0%p 조정 가능
**판촉수수료**: 제휴점 채널 기준 건당 최대 50만원
**연체이자율**: 약정이자율 + 3%p (최대 연 20%)
**슬라이딩**: 3회 이상 연체 시 등급 하향 조정
**증빙서류**: 신분증, 소득증빙, 차량등록증, 매매계약서
**HJSeg**: 차량 세그먼트별 LTV 차등 적용

### (임직원대출) 임직원대출(ESM) 운영기준
**상품명**: 임직원 중고승용차 구입대출
**대상**: JB우리캐피탈 임직원 및 등록 딜러
**금리**: 임직원 우대금리 적용 (일반 대비 -2.0%p)
**NICE등급 기준**: 6등급 이하 취급 불가
**대출가능횟수**: 임직원 연 2회, 딜러 연 4회
**실적급수수료**: 딜러 실적에 따라 차등 지급
- A등급 딜러: 건당 30만원
- B등급 딜러: 건당 20만원
**조정금리**: 본부장 전결로 최대 -0.5%p 추가 조정
**슬라이딩**: 2회 연체 시 우대금리 해제

### (신용구제) 신용구제 상품 운영기준
**상품명**: 신용구제 대상 중고승용차 할부
**대상**: 개인회생/신용회복 완료 고객
**시범운영**: 월 취급액 한도 5억원 (초과 시 본부 승인)
**연령제한**: 만 25세 ~ 65세
**상환방식**: 원리금균등분할상환만 가능
**금리**: 고정금리 연 12.9% ~ 15.9%
**대출기간**: 최대 60개월
**조정금리**: 불가 (고정금리 상품)
**판촉수수료**: 제휴점 채널 한정, 건당 최대 30만원

### (Dual Offer) Dual Offer 운영기준
**상품명**: 중고승용차 Dual Offer 구입대출
**상품구분**: 한도확대 + 금리상향 패키지
**세부코드**: DUAL-01 (일반), DUAL-02 (프리미엄)
**최대개월수**: 96개월
**대상**: 내국인/외국인 (NICE 1~10등급)
**최저금리**: 연 8.9% (1등급 기준)
**판촉수수료**: 건당 최대 70만원
**중도상환수수료율**: 잔여원금의 1.5% (2년 이내 상환 시)
**슬라이딩**: 일반 론/할부와 동일 기준 적용

### (엔카) 엔카 다이렉트 운영기준
**상품명**: 엔카 무수수료 중고차 구입대출
**플랫폼**: 엔카 다이렉트
**대상**: 엔카 상담조회 이력 보유 고객
**수수료**: 무수수료 (판촉수수료 없음)
**금리등급**: 1~12등급 (NICE 기준)
**금리**: 론/할부 대비 +0.3%p 가산
**NEGO**: 불가 (고정 가산금리)
**연체이자율**: 약정이자율 + 3%p
**중도상환수수료**: 잔여원금의 1.0%
**실적급수수료**: 엔카 플랫폼에 건당 10만원 지급
""",

    "전략금융 운영기준 #text": """
### (재고금융) 재고금융 운영기준
**상품명**: 중고차 재고금융
**대상**: 중고차 매매상사 (사업자등록 6개월 이상)
**담보**: 재고 차량 (실사 기반)
**한도산정**: 재고차량 감정가의 70% 이내
**등급관리**: 연 1회 정기실사, 분기별 서류점검
**약정서**: 포괄근담보 약정 (갱신형)
**연체 시**: 3개월 이상 연체 시 담보 처분 개시
**사후관리**: 월별 재고현황 보고서 징구, 분기별 현장실사
**이자율**: 기본 연 6.5% ~ 9.5% (등급별 차등)
**LTV**: 일반 중고차 70%, 수입차 60%

### (제휴점 운영자금) 제휴점 운영자금 운영기준
**상품명**: 제휴점 운영자금 대출
**대상**: JB우리캐피탈 등록 제휴점
**한도**: 월 매출 규모 기반 평가 (최대 3억원)
**목표달성**: 분기별 실적 목표 미달 시 한도 축소
**패널티**: 2분기 연속 미달 시 운영자금 회수 검토
**이자율**: 연 5.5% ~ 7.5%
**상환방식**: 일시상환 또는 분할상환

### (매매상사 운영자금) 매매상사 운영자금 운영기준
**상품명**: 매매상사 운영자금 대출
**대상**: 중고차 매매상사
**한도산정**: 최근 3개월 최고 판매월의 판매대수 × 단가
**NICE CB 기준**: 사업자 CB 700점 이상
**영업본부장 전결**: 1억원 초과 시 영업본부장 승인 필요
**이자율**: 연 6.0% ~ 8.5%
**기간**: 최대 12개월 (갱신 가능)

### (운영자금 자금용도 기준) 운영자금 자금용도별 기준
**자금용도 분류**:
- 차량구입: 매매계약서, 차량등록증 사본
- 타사대환: 기존 대출 상환내역서, 대환확인서
- 시설투자: 견적서, 세금계산서
- 운전자금: 자금사용계획서
**증빙방법**: 대출 실행 후 30일 이내 증빙 제출
**미이행 시**: 기한이익상실 사유 해당, 전액 즉시 상환 요구 가능
**패널티**: 미증빙 시 약정이자율 +2%p 가산

### (임차보증금) 임차보증금 운영기준
**상품명**: 임차보증금 대출
**대상**: 매매상사 사업장 임차 사업자
**금리**: 연 5.0% ~ 7.0% (신용등급별)
**한도**: 임차보증금의 80% 이내 (최대 2억원)
**기간**: 임대차계약 잔여기간 이내
**상환방식**: 만기일시상환
""",

    "중고리스 운영기준 #text": """
### (중고리스) 중고리스 운영기준
**상품명**: 중고차 운용리스 / 금융리스
**리스유형**:
- 운용리스: 잔가 설정, 만기 시 반납/매수/재리스 선택
- 금융리스: 잔가 없음, 만기 시 소유권 이전
**잔가율**: 차종별 차등 적용
- 그룹 I (국산 세단): 잔가율 30~40%
- 그룹 II (국산 SUV): 잔가율 35~45%
- 그룹 III-P (수입차 프리미엄): 잔가율 25~35%
**대상차량**: 출고 후 7년 이내, 주행거리 15만km 이내
**수입차 특칙**: 공식 딜러 인증 중고차만 취급
**보험**: 리스기간 중 종합보험 의무가입
**세금**: 운용리스는 부가세 매입세액 공제 가능
**이자율**: 운용리스 연 5.9%~8.9%, 금융리스 연 6.5%~9.5%
**기간**: 최소 24개월 ~ 최대 60개월
""",

    "중형트럭 운영기준 #text": """
### (중형트럭) 중형트럭 할부/대출 운영기준
**상품명**: 중형트럭 구입 할부/대출
**대상차량**: 총중량 3.5톤 ~ 8톤 중형트럭
**특장차량**: 냉동탑차, 윙바디, 카고크레인 등 특장 포함
**LTV**: 일반 중형트럭 80%, 특장차량 70%
**주행거리 기준**: 연간 6만km 이내 차량
**예외협의**: 주행거리 초과 차량은 심사역 개별 심사
**대출기간**: 최소 12개월 ~ 최대 72개월
**금리**: 연 7.5% ~ 11.5% (신용등급 및 차령별)
**대상고객**: 개인사업자, 법인 (운송업 등록)
**증빙서류**: 사업자등록증, 운송사업 면허증, 차량등록증
**판촉수수료**: 제휴점 채널 건당 최대 40만원
""",
}

# 이미지 더미 데이터 (중고승용만 has_image=True)
DUMMY_IMAGE_DOCS = [
    {
        "title": "중고승용 운영기준 이미지 #image",
        "content": "질문 : 론/할부 운영기준 이미지",
        "metadata": {
            "title": "론/할부 운영기준 이미지 #image",
            "질문": "론/할부 운영기준 이미지 #image",
            "답변": "론/할부 운영기준 요약 이미지입니다.",
            "source": "http://localhost:3000",
            "html_images": "![Image](https://via.placeholder.com/600x300?text=론/할부+운영기준)",
        },
    },
    {
        "title": "임직원대출 운영기준 이미지 #image",
        "content": "질문 : 임직원대출(ESM) 운영기준 이미지",
        "metadata": {
            "title": "임직원대출 운영기준 이미지 #image",
            "질문": "ESM, 딜러 전용 대출(딜러 대출) 상품 운영기준 알려줘 #image",
            "답변": "임직원대출(ESM) 운영기준 요약 이미지입니다.",
            "source": "http://localhost:3000",
            "html_images": "![Image](https://via.placeholder.com/600x300?text=임직원대출+운영기준)",
        },
    },
]


# ============================================================
# ChromaDB OpenAI 임베딩 함수
# ============================================================
class OpenAIEmbeddingFunction:
    """chromadb 호환 OpenAI 임베딩 함수"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def name(self) -> str:
        return "openai_embedding"

    def __call__(self, input: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=input)
        return [r.embedding for r in resp.data]

    def embed_query(self, input: list[str]):
        embeddings = self.__call__(input)
        return [np.array(e) for e in embeddings]


# ============================================================
# 텍스트 전처리 (기존 파이프라인과 동일 로직)
# ============================================================
def split_text_by_separator(text: str, separator: str = "###") -> list[str]:
    chunks = []
    for part in text.split(separator):
        cleaned = part.strip()
        if cleaned and cleaned != "\\":
            chunks.append(cleaned)
    return chunks


def text_doc_preprocesser(text: str) -> list[dict]:
    """기존 파이프라인의 text_doc_preprocesser 로직 재현"""
    docs = []
    chunks = split_text_by_separator(text)
    for chunk in chunks:
        pattern = r"\((.*?)\)"
        match = re.search(pattern, chunk)
        if match:
            subtask = match.group(0)  # e.g., "(론/할부)"
            docs.append({"page_content": chunk, "subtask": subtask})
    return docs


# ============================================================
# Outline wiki 데이터 생성 (선택)
# ============================================================
def setup_outline(outline_url: str, api_key: str):
    """Outline wiki에 컬렉션 + 더미 문서 생성"""
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 1. 컬렉션 생성
    print("[Outline] 컬렉션 생성 중...")
    resp = requests.post(
        f"{outline_url}/api/collections.create",
        json={"name": "오토운영팀 운영기준", "permission": "read_write"},
        headers=headers,
    )
    if resp.status_code != 200:
        print(f"  컬렉션 생성 실패 (이미 존재할 수 있음): {resp.status_code}")
        # 기존 컬렉션 검색
        resp2 = requests.post(
            f"{outline_url}/api/collections.list",
            json={},
            headers=headers,
        )
        collections = resp2.json().get("data", [])
        collection_id = None
        for c in collections:
            if "오토운영팀" in c.get("name", ""):
                collection_id = c["id"]
                break
        if not collection_id:
            print("  기존 컬렉션도 찾을 수 없음. Outline 설정을 건너뜁니다.")
            return
    else:
        collection_id = resp.json()["data"]["id"]

    print(f"  컬렉션 ID: {collection_id}")

    # 2. 문서 생성
    for title, text in DUMMY_TEXT_DOCS.items():
        print(f"  문서 생성: {title}")
        resp = requests.post(
            f"{outline_url}/api/documents.create",
            json={
                "title": title,
                "text": text.strip(),
                "collectionId": collection_id,
                "publish": True,
            },
            headers=headers,
        )
        if resp.status_code != 200:
            print(f"    실패: {resp.status_code} - {resp.text[:200]}")

    print("[Outline] 더미 문서 생성 완료!")
    return collection_id


# ============================================================
# ChromaDB 데이터 삽입
# ============================================================
def setup_chromadb(api_key: str, chroma_host: str, chroma_port: int):
    """ChromaDB에 더미 데이터를 OpenAI 임베딩으로 삽입"""
    print(f"\n[ChromaDB] {chroma_host}:{chroma_port} 연결 중...")
    chroma = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    embed_fn = OpenAIEmbeddingFunction(api_key=api_key, model=OPENAI_EMBEDDING_MODEL)

    # --- 텍스트 컬렉션 ---
    print(f"[ChromaDB] '{TEXT_COLLECTION}' 컬렉션 생성...")
    try:
        chroma.delete_collection(TEXT_COLLECTION)
    except Exception:
        pass

    text_col = chroma.get_or_create_collection(
        name=TEXT_COLLECTION,
        embedding_function=embed_fn,
    )

    # 모든 #text 문서를 처리
    all_docs = []
    for title, raw_text in DUMMY_TEXT_DOCS.items():
        docs = text_doc_preprocesser(raw_text)
        all_docs.extend(docs)

    print(f"  텍스트 청크 수: {len(all_docs)}")

    text_col.upsert(
        documents=[d["page_content"] for d in all_docs],
        ids=[f"{TEXT_COLLECTION}{i}" for i in range(len(all_docs))],
        metadatas=[{"subtask": d["subtask"]} for d in all_docs],
    )
    print(f"  '{TEXT_COLLECTION}' 삽입 완료!")

    # --- 이미지 컬렉션 ---
    print(f"[ChromaDB] '{IMAGE_COLLECTION}' 컬렉션 생성...")
    try:
        chroma.delete_collection(IMAGE_COLLECTION)
    except Exception:
        pass

    image_col = chroma.get_or_create_collection(
        name=IMAGE_COLLECTION,
        embedding_function=embed_fn,
    )

    image_col.upsert(
        documents=[d["content"] for d in DUMMY_IMAGE_DOCS],
        ids=[f"{IMAGE_COLLECTION}{i}" for i in range(len(DUMMY_IMAGE_DOCS))],
        metadatas=[d["metadata"] for d in DUMMY_IMAGE_DOCS],
    )
    print(f"  '{IMAGE_COLLECTION}' 삽입 완료! ({len(DUMMY_IMAGE_DOCS)}건)")

    # 검증
    print("\n[검증] 텍스트 컬렉션 조회...")
    result = text_col.query(query_texts=["론/할부 금리"], n_results=3)
    for i, doc in enumerate(result["documents"][0]):
        subtask = result["metadatas"][0][i].get("subtask", "?")
        print(f"  [{i+1}] subtask={subtask}, content={doc[:80]}...")

    print("\n[완료] 모든 더미 데이터 삽입 완료!")


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="더미 데이터 생성 스크립트")
    parser.add_argument("--openai-api-key", required=True, help="OpenAI API 키")
    parser.add_argument("--chroma-host", default="localhost", help="ChromaDB 호스트")
    parser.add_argument("--chroma-port", type=int, default=8800, help="ChromaDB 포트")
    parser.add_argument("--outline-url", default="", help="Outline wiki URL (비워두면 건너뜀)")
    parser.add_argument("--outline-api-key", default="", help="Outline API 키")
    args = parser.parse_args()

    # Outline wiki 설정 (선택)
    if args.outline_url and args.outline_api_key:
        setup_outline(args.outline_url, args.outline_api_key)
    else:
        print("[Outline] URL/API키 미제공 → Outline 설정 건너뜀")

    # ChromaDB 설정 (필수)
    setup_chromadb(args.openai_api_key, args.chroma_host, args.chroma_port)


if __name__ == "__main__":
    main()
