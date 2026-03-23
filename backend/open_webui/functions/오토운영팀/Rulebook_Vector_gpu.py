from pydantic import BaseModel, Field
from typing import Union, Generator, Iterator
import sys
import os
import subprocess
import tempfile
import gc
import time
import select


class Pipe:
    class Valves(BaseModel):
        # --- [설정값] ---
        BASE_IMG_URL: str = "https://ai.wooricap.com/static/webrule_wiki_images/"
        STATIC_IMG_URL: str = (
            "/home/ubuntu/anaconda3/envs/webui/lib/python3.11/site-packages/open_webui/static/webrule_wiki_images/"
        )
        EMBED_PATH: str = "/data1/embedding/kure"
        CHROMA_PORT: int = 8800
        CHROMA_IP: str = "172.18.237.81"
        CHROMA_COLLECTION_NAME: str = "webrule_wiki_gpu"
        OUTLINE_API_KEY: str = "ol_api_QAwIu3iE5cyRFVLJBT3uB15iJD6n8KXvqU7WCh"
        OUTLINE_URL: str = "https://util.wooricap.com:7443"
        COLLECTION_ID: str = "c6472f73-b9af-48a3-945e-8a16320f4ef6"

        # [안전장치 설정]
        MAX_WORKERS: int = Field(
            default=2, description="최대 동시 실행 워커 수 (안전값: 2)"
        )
        EMBED_BATCH_SIZE: int = Field(
            default=16, description="배치 사이즈 (OOM 방지를 위해 작게 설정)"
        )
        WORKER_START_DELAY: float = Field(
            default=3.0, description="워커 실행 간격(초) - 메모리 스파이크 방지"
        )
        # MIG_UUIDS: str = Field(default="MIG-5770d9ff-2108-56a3-80f4-79a1209cbf47,MIG-cd835075-7587-5509-993f-0da938b44b90,MIG-dc63a65d-2530-5358-956f-532938402bdc,MIG-21759f22-7985-5542-8dd7-1efb1f3e7b52")
        MIG_UUIDS: str = Field(
            default="MIG-dc63a65d-2530-5358-956f-532938402bdc,MIG-21759f22-7985-5542-8dd7-1efb1f3e7b52"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _read_process_output(self, process, buffer: str) -> tuple[str, list[str]]:
        """
        Non-blocking으로 프로세스 출력을 읽고, 완성된 라인들을 반환
        Returns: (남은 버퍼, 완성된 라인 리스트)
        """
        lines = []
        try:
            chunk = process.stdout.read(4096)
            if chunk:
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    lines.append(line + "\n")
        except BlockingIOError:
            pass
        except Exception:
            pass
        return buffer, lines

    def _wait_with_yield(
        self, seconds: float, message_prefix: str = ""
    ) -> Generator[str, None, None]:
        """긴 sleep을 짧게 분할하여 Generator가 블로킹되지 않도록 함"""
        intervals = int(seconds * 10)  # 0.1초 단위
        for i in range(intervals):
            time.sleep(0.1)
            # 매 초마다 진행 상황 표시 (선택적)
            if message_prefix and (i + 1) % 10 == 0:
                elapsed = (i + 1) / 10
                yield f"{message_prefix} ({elapsed:.1f}s / {seconds:.1f}s)\n"

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        temp_files = []
        all_processes = []  # 모든 프로세스 추적 (에러 체크용)
        active_processes = []  # 현재 실행 중인 프로세스
        process_buffers = {}  # 프로세스별 출력 버퍼
        data_file_path = None

        try:
            yield "\n🚀 [시스템] 파이프라인 가동\n"
            # ==================================================================
            # [Step 1] 리소스 확보
            # ==================================================================
            configured_migs = [
                uuid.strip()
                for uuid in self.valves.MIG_UUIDS.split(",")
                if uuid.strip()
            ]

            if not configured_migs:
                yield f"⚠️ MIG가 설정되지 않음.\n"
                return

            safe_max_workers = min(self.valves.MAX_WORKERS, len(configured_migs))
            target_migs = configured_migs[:safe_max_workers]
            gpu_count = len(target_migs)

            yield f"✅ 할당된 MIG: {gpu_count}개\n"

            # ==================================================================
            # [Step 2] 워커 스크립트 생성 (JSON Lines 방식 적용)
            # ==================================================================
            worker_script = '''
import sys
import time
import uuid
import json
import os
import gc
import requests
import html
import re
import math
import itertools
from typing import List, Dict
from urllib.parse import urlparse, parse_qs
from langchain.schema import Document

Cell = Dict[str, object]

class MockValves:
    def __init__(self):
        self.BASE_IMG_URL = os.environ.get("BASE_IMG_URL", "https://ai.wooricap.com/static/webrule_wiki_images_jjw_test/")
        self.EMBED_PATH = os.environ.get("EMBED_PATH", "/data1/embedding/kure")
        self.CHROMA_IP = os.environ.get("CHROMA_IP", "172.18.237.81")
        self.CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8800))
        self.CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "webrule_wiki_gpu_jjw_test")
        self.COLLECTION_ID = os.environ.get("COLLECTION_ID", "c6472f73-b9af-48a3-945e-8a16320f4ef6")
        self.EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", 16))
        self.OUTLINE_API_KEY = os.environ.get("OUTLINE_API_KEY", "ol_api_QAwIu3iE5cyRFVLJBT3uB15iJD6n8KXvqU7WCh")
        self.OUTLINE_URL = os.environ.get("OUTLINE_URL", "https://util.wooricap.com:7443")
        self.STATIC_IMG_URL = os.environ.get("STATIC_IMG_URL", "/home/ubuntu/anaconda3/envs/webui/lib/python3.11/site-packages/open_webui/static/webrule_wiki_images_jjw_test/")


class WorkerProcessor:
    def __init__(self):
        self.valves = MockValves()

    def get_wiki_documents(self, collectionId: str):
        os.makedirs(self.valves.STATIC_IMG_URL, exist_ok=True)
        self.delete_all_files_in_directory(self.valves.STATIC_IMG_URL)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.OUTLINE_API_KEY}",
        }
        docs = []
        json = {"offset": 0, "limit": 1, "collectionId": collectionId}

        check_total = requests.post(
            f"{self.valves.OUTLINE_URL}/api/documents.list",
            json=json,
            headers=headers,
        )

        if not check_total.ok:
            raise ValueError("Outline API returned an error: ", check_total.text)

        # 전체 조회 페이지 계산
        total = check_total.json()["pagination"]["total"]
        total_page = math.ceil(total / 100)

        # 페이지별 조회. 한번에 최대 조회 100개씩만 가능
        for page in range(total_page):
            offset = page * 100
            json = {
                "offset": offset,
                "limit": 100,
                "collectionId": collectionId,
                "sort": "updatedAt",
                "direction": "ASC",
            }

            raw_result = requests.post(
                f"{self.valves.OUTLINE_URL}/api/documents.list",
                json=json,
                headers=headers,
            )
            results = raw_result.json()["data"]

            for result in results:
                # 웹규정집은 체크 표시가 있는 문서만 유효
                if collectionId == "c6472f73-b9af-48a3-945e-8a16320f4ef6":
                    if result["icon"] != "✔️":
                        continue

                doc_id = result["id"]

                detail_result = requests.post(
                    f"{self.valves.OUTLINE_URL}/api/documents.info",
                    json={"id": doc_id},
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.valves.OUTLINE_API_KEY}",
                        "Accept": f"application/json",
                        "x-api-version": "3",
                    },
                )
                dr = detail_result.json()["data"]["document"]
                json_content = detail_result.json()["data"]["document"]["data"]
                # 복잡한표는 html, 나머지 마크다운
                md_convert = self.doc_to_md(json_content)
                md_convert = md_convert.replace("\\xa0", " ")

                urls = self.extract_image_urls(md_convert)
                modified_text = md_convert

                for url in urls:
                    image_id = self.extract_id(url)
                    file_name = self.download_image(url, dr["id"], image_id)

                    # 이미지 url을 태그 처리
                    # 텍스트에서 링크를 <ImageInWiki> 태그로 변경
                    modified_text = self.replace_image_url_to_tag(
                        modified_text, file_name
                    )

                    # html 형태 이미지 태그로 변경
                    modified_text = self.process_content_with_images(modified_text)

                metadata = {
                    "id": dr["id"],
                    "title": dr["title"],
                    "source": self.valves.OUTLINE_URL + dr["url"],
                    "text": modified_text,
                    "collection_id": dr["collectionId"],
                    "parent_document_id": dr["parentDocumentId"],
                    # "revision": dr["revision"],
                    "created_by": dr["createdBy"]["name"],
                    "updated_by": dr["updatedBy"]["name"],
                    "created_at": dr["createdAt"],
                    "updated_at": dr["updatedAt"],
                }

                doc = Document(
                    page_content="",
                    metadata={**metadata},
                )
                docs.append(doc)

            del results
            del raw_result

        all_documents = []
        for doc in docs:
            all_text = doc.metadata["text"]
            structure = self.make_structure(all_text)
            if not structure:
                another_docs = self.process_another(doc.metadata)
                all_documents.extend(another_docs)
            else:
                chapters, etc_sections = self.classify_sections(structure)
                chapters_docs = self.process_chapter_section(chapters, doc.metadata)
                etc_sections_docs = self.process_etc(etc_sections, doc.metadata)
                all_documents.extend(chapters_docs)
                all_documents.extend(etc_sections_docs)

        return all_documents


    def clean_markdown_for_search(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\\\\n", "\\n", text)

        text = re.sub(r"\\\\", "", text)

        # 헤더 처리
        text = re.sub(r"^(#{1,6})\\s+(.+)$", r"\\2", text, flags=re.MULTILINE)

        # 마크다운 링크 처리 [텍스트](링크) -> 텍스트
        text = re.sub(r"\\[([^\\]]+)\\]\\([^)]+\\)", r"\\1", text)

        # 이미지 링크 처리 ![대체텍스트](이미지링크) -> 대체텍스트
        text = re.sub(r"!\\[.*?\\]\\(.*?\\)", "", text)

        # 강조 문법 제거 (**텍스트** -> 텍스트)
        text = re.sub(r"\\*\\*([^*]+)\\*\\*", r"\\1", text)
        text = re.sub(r"__([^_]+)__", r"\\1", text)

        # 기울임 문법 제거 (*텍스트* -> 텍스트)
        text = re.sub(r"\\*([^*]+)\\*", r"\\1", text)
        text = re.sub(r"_([^_]+)_", r"\\1", text)

        # 취소선 제거 (~~텍스트~~ -> 텍스트)
        text = re.sub(r"~~([^~]+)~~", r"\\1", text)

        # 수평선 제거 (---, ***, ___)
        text = re.sub(r"^(---|\\*\\*\\*|___)$", "", text, flags=re.MULTILINE)

        # HTML 태그 제거
        text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)

        # 순서 없는 목록 기호 제거 (*, -, +)
        text = re.sub(r"^\\s*[-*+]\\s+", "", text, flags=re.MULTILINE)

        # 순서 있는 목록 번호 제거 (1., 2., 등)
        text = re.sub(r"^\\s*\\d+\\.\\s+", "", text, flags=re.MULTILINE)

        # 표 구분자 행 제거 - 개선된 정규식
        text = re.sub(
            r"^\\s*\\|[-\\s:]*(?:\\|[-\\s:]*)+\\|\\s*$", "", text, flags=re.MULTILINE
        )

        # 표 처리 - 각 행별로 처리하여 정확도 향상
        lines = text.split("\\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("|") and line.strip().endswith("|"):
                # 파이프로 분리하고 앞뒤 공백 제거
                cells = [cell.strip() for cell in line.strip()[1:-1].split("|")]
                # 빈 셀 처리 (연속 파이프 || 처리)
                cells = [cell if cell else "" for cell in cells]
                # 셀을 공백으로 구분하여 재결합
                lines[i] = " ".join(cells)
        text = "\\n".join(lines)

        # 코드 블록 처리
        text = re.sub(
            r"```[\\s\\S]*?```", lambda m: m.group(0).replace("```", "").strip(), text
        )

        # 줄바꿈을 보존하면서 연속된 공백 정리
        text = re.sub(r"[^\\S\\n]+", " ", text)

        # 각 줄의 앞뒤 공백 제거
        lines = [line.strip() for line in text.split("\\n")]
        text = "\\n".join(lines)

        # 연속된 빈 줄 정리 (3개 이상의 줄바꿈을 2개로)
        text = re.sub(r"\\n{3,}", "\\n\\n", text)

        return text.strip()

    def clean_markdown_for_llm(self, text: str) -> str:
        # 표처리 제외
        if not text:
            return ""

        text = re.sub(r"\\\\n", "\\n", text)

        text = re.sub(r"\\\\", "", text)

        # 헤더 처리
        # text = re.sub(r"^(#{1,6})\\s+(.+)$", r"\\2", text, flags=re.MULTILINE)

        # 마크다운 링크 처리 [텍스트](링크) -> 텍스트
        # text = re.sub(r"\\[([^\\]]+)\\]\\([^)]+\\)", r"\\1", text)

        # 이미지 링크 처리 ![대체텍스트](이미지링크) -> 대체텍스트
        # text = re.sub(r"!\\[.*?\\]\\(.*?\\)", "", text)

        # 강조 문법 제거 (**텍스트** -> 텍스트)
        text = re.sub(r"\\*\\*([^*]+)\\*\\*", r"\\1", text)
        text = re.sub(r"__([^_]+)__", r"\\1", text)

        # 기울임 문법 제거 (*텍스트* -> 텍스트)
        text = re.sub(r"\\*([^*]+)\\*", r"\\1", text)
        text = re.sub(r"_([^_]+)_", r"\\1", text)

        # 취소선 제거 (~~텍스트~~ -> 텍스트)
        text = re.sub(r"~~([^~]+)~~", r"\\1", text)

        # 수평선 제거 (---, ***, ___)
        text = re.sub(r"^(---|\\*\\*\\*|___)$", "", text, flags=re.MULTILINE)

        # HTML 태그 제거
        # text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)

        # 순서 없는 목록 기호 제거 (*, -, +)
        # text = re.sub(r'^\\s*[-*+]\\s+', '', text, flags=re.MULTILINE)

        # 순서 있는 목록 번호 제거 (1., 2., 등)
        # text = re.sub(r'^\\s*\\d+\\.\\s+', '', text, flags=re.MULTILINE)

        # 코드 블록 처리
        text = re.sub(
            r"```[\\s\\S]*?```", lambda m: m.group(0).replace("```", "").strip(), text
        )

        # 줄바꿈을 보존하면서 연속된 공백 정리
        text = re.sub(r"[^\\S\\n]+", " ", text)

        # 각 줄의 앞뒤 공백 제거
        lines = [line.strip() for line in text.split("\\n")]
        text = "\\n".join(lines)

        # 연속된 빈 줄 정리 (3개 이상의 줄바꿈을 2개로)
        text = re.sub(r"\\n{3,}", "\\n\\n", text)

        return text.strip()

    def make_structure(self, text: str):
        # 공백과 마크다운 헤더를 모두 고려한 패턴
        chapter_pattern = r"(?:^|\\n)\\s*#{0,4}\\s*제\\s*(\\d+)\\s*장\\s*([^\\n]*)"
        section_pattern = r"(?:^|\\n)\\s*#{0,4}\\s*제\\s*(\\d+)\\s*절\\s*([^\\n]*)"
        article_pattern = r"(?:^|\\n)\\s*#{0,4}\\s*제\\s*(\\d+)\\s*조\\s*(?:\\(([^)]*)\\))?\\s*(.*?)(?=(?:(?:^|\\n)\\s*#{0,4}\\s*제\\s*\\d+\\s*(?:장|절|조))|(?:(?:^|\\n)\\s*#{0,4}\\s*<(?:별표|별지서식))|(?:\\n\\s*부칙)|$)"
        annexes_and_forms_pattern = r"(?:^|\\n)\\s*#{0,4}\\s*<(별표|별지서식)\\s*(\\d*)>([\\s\\S]*?)(?=\\n\\s*#{0,4}\\s*<(별표|별지서식)\\s*\\d*>|$)"

        structure = []
        current_chapter = None
        current_section = None

        # 먼저 전체 텍스트에서 모든 매치를 찾아서 위치 순으로 정렬
        all_matches = []

        # 장 매치
        for match in re.finditer(chapter_pattern, text):
            all_matches.append(("chapter", match.start(), match))

        # 절 매치
        for match in re.finditer(section_pattern, text):
            all_matches.append(("section", match.start(), match))

        # 조 매치
        for match in re.finditer(article_pattern, text, re.DOTALL):
            all_matches.append(("article", match.start(), match))

        # 위치순으로 정렬
        all_matches.sort(key=lambda x: x[1])

        for match_type, pos, match in all_matches:
            if match_type == "chapter":
                chapter_num = match.group(1)
                chapter_title = match.group(2)
                current_chapter = {
                    "chapter": f"제{chapter_num}장",
                    "title": chapter_title.strip(),
                    "sections": [],
                }
                structure.append(current_chapter)
                current_section = None

            elif match_type == "section":
                section_num = match.group(1)
                section_title = match.group(2)
                current_section = {
                    "section": f"제{section_num}절",
                    "title": section_title.strip(),
                    "articles": [],
                }
                if current_chapter:
                    current_chapter["sections"].append(current_section)

            elif match_type == "article":
                article_number = match.group(1)
                article_title = match.group(2) if match.group(2) else ""
                article_content = match.group(3) if match.group(3) else ""

                # 장이 없는 경우 기본 장 생성
                if current_chapter is None:
                    current_chapter = {
                        "chapter": "",
                        "title": "",
                        "sections": [],
                    }
                    structure.append(current_chapter)

                article = {
                    "article_number": f"제{article_number}조",
                    "title": article_title.strip() if article_title else None,
                    "content": [
                        line.strip()
                        for line in article_content.splitlines()
                        if line.strip()
                    ],
                }

                # 절이 있으면 절에 추가, 없으면 장에 직접 추가
                if current_section:
                    current_section["articles"].append(article)
                else:
                    current_chapter["sections"].append(
                        {"section": None, "articles": [article]}
                    )

        # 별표 및 별지서식 탐색
        for match in re.finditer(annexes_and_forms_pattern, text, re.DOTALL):
            item_type = match.group(1)
            number = match.group(2)
            content = match.group(3).strip()  # 콘텐츠 추출

            item = {
                "etc_title": f"<{item_type}{number}>",
                "etc_content": [
                    line.strip() for line in content.splitlines() if line.strip()
                ],
            }

            structure.append(item)

        return structure

    # 장절조 process
    def process_chapter_section(self, structure, meta):
        documents = []
        chapter_merge = []
        for chapter in structure:
            chapter_number = chapter.get("chapter", "")
            chapter_title = chapter.get("title", "")

            for section in chapter.get("sections", []):
                section_number = section.get("section", "")
                section_title = section.get("title", "")

                # 절이 없는 경우 빈 값으로 처리
                if not section_number:
                    section_number = ""
                if not section_title:
                    section_title = ""

                for article in section.get("articles", []):
                    article_number = article["article_number"]
                    article_title = article.get("title", "")
                    article_content = "\\n".join(article["content"])
                    # page_content = f"{meta['title']}\\n{chapter_number} {chapter_title} \\n {article_number} ({article_title}) {article_content.strip()}"
                    page_content = f"{chapter_number} {chapter_title}\\n{article_number} ({article_title}) {article_content.strip()}"

                    # 메타데이터 구성
                    metadata = {
                        "chapter": chapter_number,
                        "chapter_title": chapter_title,
                        "section": section_number,
                        "section_title": section_title,
                        "article_number": article_number,
                        "article_title": article_title,
                        "id": meta["id"],
                        "title": meta["title"],
                        "source": meta["source"],
                        "text": self.clean_markdown_for_llm(article_content.strip()),
                        "collection_id": meta["collection_id"],
                        "parent_document_id": meta["parent_document_id"],
                        "created_by": meta["created_by"],
                        "updated_by": meta["updated_by"],
                        "created_at": meta["created_at"],
                        "updated_at": meta["updated_at"],
                    }

                    # Document 객체 생성 및 추가
                    documents.append(
                        Document(
                            page_content=self.clean_markdown_for_search(page_content),
                            metadata=metadata,
                        )
                    )
        return documents

    # 별첨 process
    def process_etc(self, structure, meta):
        documents = []
        for etc in structure:
            etc_content = etc.get("etc_content", "")
            etc_title = etc.get("etc_title", "")
            if etc_title:
                etc_content = "\\n".join(etc_content)
                metadata = {
                    "etc_title": etc_title,
                    "etc_name": etc_content.split("\\n")[0],
                    "id": meta["id"],
                    "title": meta["title"],
                    "source": meta["source"],
                    "text": self.clean_markdown_for_llm(etc_content),
                    "collection_id": meta["collection_id"],
                    "parent_document_id": meta["parent_document_id"],
                    "created_by": meta["created_by"],
                    "updated_by": meta["updated_by"],
                    "created_at": meta["created_at"],
                    "updated_at": meta["updated_at"],
                }

                documents.append(
                    Document(
                        page_content=self.clean_markdown_for_search(etc_content),
                        metadata=metadata,
                    )
                )

        return documents

    # 장/절/조 없는 파일 [전체 저장]
    def process_another(self, meta):
        documents = []
        text = meta["text"]
        metadata = {
            "id": meta["id"],
            "title": meta["title"],
            "source": meta["source"],
            "text": self.clean_markdown_for_llm(text),
            "collection_id": meta["collection_id"],
            "parent_document_id": meta["parent_document_id"],
            "created_by": meta["created_by"],
            "updated_by": meta["updated_by"],
            "created_at": meta["created_at"],
            "updated_at": meta["updated_at"],
        }

        documents.append(
            Document(
                page_content=self.clean_markdown_for_search(text),
                metadata=metadata,
            )
        )

        return documents

    # structure 구분 기능
    def classify_sections(self, structure):
        chapters = []
        etc_sections = []
        for section in structure:
            if (
                "chapter" in section.keys()
            ):  # Check if the dictionary has a 'chapter' key
                chapters.append(section)
            elif (
                "etc_title" in section.keys()
            ):  # Check if the dictionary has an 'etc_title' key
                etc_sections.append(section)

        return chapters, etc_sections

    def plain_text(self, node) -> str:
        if isinstance(node, list):
            return "".join(self.plain_text(child) for child in node)
        if not isinstance(node, dict):
            return ""

        node_type = node.get("type")
        if "text" in node:
            return node["text"]

        if node_type == "hard_break":
            return "\\n"

        if node_type == "br":
            return "\\n"

        if node_type == "image":
            return f"![]({node['attrs']['src']})"

        body = "".join(self.plain_text(child) for child in node.get("content", []))

        if node_type == "paragraph":
            return body + "\\n"
        return body

    def has_merge(self, rows: List[List[Cell]]) -> bool:
        return any(
            c.get("colspan", 1) > 1 or c.get("rowspan", 1) > 1
            for c in itertools.chain.from_iterable(rows)
        )

    def normalize_matrix(self, rows: List[List[Cell]]) -> List[List[Cell]]:
        grid, reserves = [], {}
        r_idx = 0

        for row in rows:
            cur, col = [], 0
            while True:
                if (r_idx, col) in reserves:
                    cur.append({"text": "", "is_header": False})
                    reserves[(r_idx, col)] -= 1
                    if reserves[(r_idx, col)] == 0:
                        del reserves[(r_idx, col)]
                    col += 1
                    continue

                if not row:
                    break

                cell = row.pop(0)
                cs, rs = cell.get("colspan", 1), cell.get("rowspan", 1)
                cur.append(cell)

                for _ in range(1, cs):
                    cur.append({"text": "", "is_header": False})

                for i in range(1, rs):
                    for j in range(cs):
                        # reserves[(r_idx + i, col + j)] = rs - i
                        reserves[(r_idx + i, col + j)] = reserves[
                            (r_idx + i, col + j), 0
                        ]

                col += cs

            grid.append(cur)
            r_idx += 1

        width = max((len(r) for r in grid), default=0)

        for r in grid:
            r.extend([{"text": "", "is_header": False}] * (width - len(r)))

        return grid

    def table_to_html(self, rows: List[List[Cell]]) -> str:
        out = ["<table>"]

        for r in rows:
            out.append("  <tr>")
            for c in r:
                tag = "th" if c.get("is_header") else "td"
                cs = c.get("colspan", 1)
                rs = c.get("rowspan", 1)
                attr = (f' colspan="{cs}"' if cs > 1 else "") + (
                    f' rowspan="{rs}"' if rs > 1 else ""
                )

                out.append(f'    <{tag}{attr}>{html.escape(str(c["text"]))}</{tag}>')

            out.append("  </tr>")

        out.append("</table>")

        return "\\n".join(out)

    def table_to_markdown(self, rows: List[List[Cell]]) -> str:
        if self.has_merge(rows):
            return self.table_to_html(rows)

        grid = self.normalize_matrix(rows)
        widths = [max(len(str(c["text"])) for c in col) for col in zip(*grid)]

        def fmt(r):
            return (
                "| "
                + " | ".join(str(c["text"]).ljust(w) for c, w in zip(r, widths))
                + " |"
            )

        md = [fmt(grid[0]), "| " + " | ".join("-" * w for w in widths) + " |"]
        md += [fmt(r) for r in grid[1:]]

        return "\\n".join(md)

    def pm_table_to_rows(self, tbl: dict) -> List[List[Cell]]:
        rows = []

        for r in tbl.get("content", []):
            cells = []
            for c in r.get("content", []):
                a = c.get("attrs", {})
                cells.append(
                    {
                        "text": self.plain_text(c).rstrip("\\n"),
                        "colspan": a.get("colspan", 1),
                        "rowspan": a.get("rowspan", 1),
                        "is_header": c["type"] == "table_header",
                    }
                )

            rows.append(cells)

        return rows

    def block(self, node: dict) -> str:
        t = node["type"]

        if t == "heading":
            lvl = node.get("attrs", {}).get("level", 1)
            return "#" * lvl + " " + self.plain_text(node) + "\\n\\n"

        if t == "paragraph":
            txt = self.plain_text(node)
            return txt + "\\n\\n" if txt else ""

        if t in ("bullet_list", "ordered_list", "list_item"):
            return "".join(self.block(ch) for ch in node.get("content", []))

        if t == "table":
            rows = self.pm_table_to_rows(node)
            return self.table_to_markdown(rows) + "\\n\\n"

        return "".join(self.block(ch) for ch in node.get("content", []))

    def doc_to_md(self, doc: dict) -> str:
        return "".join(self.block(n) for n in doc["content"]).rstrip() + "\\n"


    def _extract_cell_content(self, cell) -> str:
        # 텍스트 추출
        content = cell.get_text()

        content = content.replace("\\r\\n", "<br>")
        content = content.replace("\\n", "<br>")
        content = content.replace("\\r", "<br>")

        # 연속된 <br> 정리
        while "<br><br>" in content:
            content = content.replace("<br><br>", "<br>")

        # 앞뒤 공백 및 <br> 제거
        content = content.strip()
        content = content.strip("<br>")

        return content


    def extract_image_urls(self, markdown_text):
        pattern = r"!\\[\\]\\((.*?)\\)"
        matches = re.findall(pattern, markdown_text)
        urls = []
        for match in matches:
            url = match.split()[0]
            if url.startswith("http") or url.startswith("https"):
                urls.append(url)
            else:
                urls.append(self.valves.OUTLINE_URL + url)
        return urls

    def extract_id(self, url):
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        return query_params.get("id", [None])[0]

    def download_image(self, url, document_id, image_id):
        HEADERS = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.OUTLINE_API_KEY}",
        }
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code == 200:
                # 파일명 형식: 문서ID_이미지파일명.확장자
                filename = f"{document_id}_{image_id}.png"
                filepath = os.path.join(self.valves.STATIC_IMG_URL, filename)

                with open(filepath, "wb") as f:
                    f.write(response.content)

                return filename
            else:
                return filename
        except Exception as e:
            return None

    def replace_image_url_to_tag(self, text, file_name):
        image_link_pattern = r"!\\[\\]\\((.*?)\\)"
        match = re.search(image_link_pattern, text)

        if match:
            replacement = f"<ImageInWiki>{file_name}</ImageInWiki>"
            return text.replace(match.group(0), replacement)
        else:
            return text

    def convert_image_to_url(self, image_path, image_name):
        if os.path.exists(image_path):
            # 이미지 URL 생성
            return f"![Image]({self.valves.BASE_IMG_URL}{image_name})"
        else:
            return "(이미지 파일을 찾을 수 없음)"

    def process_content_with_images(self, content):
        pattern = re.compile(r"<ImageInWiki>(.*?)</ImageInWiki>")

        def replace_image_tag(match):
            image_name = match.group(1).strip()
            image_path = os.path.join(self.valves.STATIC_IMG_URL, image_name)
            return self.convert_image_to_url(image_path, image_name)

        return pattern.sub(replace_image_tag, content)

    def delete_all_files_in_directory(self, directory_path):
        try:
            # 경로가 존재하는지 확인
            if not os.path.exists(directory_path):
                return

            # 파일 삭제
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)

                # 파일인지 확인하고 삭제
                if os.path.isfile(file_path):
                    os.remove(file_path)

        except Exception as e:
            print(f"IT FAQ delete_all_files_in_directory error : {e}")


def run_fetch_mode(output_path):
    """[Phase 1] 데이터 수집 - JSON Lines 형식으로 저장"""
    print("▶ [Phase 1] 데이터 수집 시작...", flush=True)
    processor = WorkerProcessor()

    try:
        docs = processor.get_wiki_documents(processor.valves.COLLECTION_ID)
       
        print(f"▶ [Phase 1] 수집 완료: 총 {len(docs)}개 문서", flush=True)
       
        # 2. JSON 직렬화 저장
        json_data = []
        for d in docs:
            json_data.append({"page_content": d.page_content, "metadata": d.metadata})
           
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
           
        print(f"▶ [Phase 1] 데이터 임시 파일 저장 완료", flush=True)

    except Exception as e:
        print(f"❌ [Phase 1 Error] {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_embed_mode(input_path, worker_id, total_workers):
    """[Phase 2] 임베딩 - 스트리밍 방식으로 JSON Lines 읽기"""
    # 지연 로딩 (메모리 절약)
    import torch
    import chromadb
    from sentence_transformers import SentenceTransformer

    sys.stdout.reconfigure(line_buffering=True)
    processor = WorkerProcessor()

    print(f"▶ [Worker-{worker_id}] 초기화 중 (PID: {os.getpid()})...", flush=True)

    try:
        # 1. 데이터 로드
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
           
        # 2. 내 몫만 필터링 (Sharding)
        my_docs = [d for i, d in enumerate(raw_data) if i % total_workers == worker_id]
       
        if not my_docs:
            print(f"▶ [Worker-{worker_id}] 처리할 데이터가 없습니다. 종료.", flush=True)
            return

        print(f"▶ [Worker-{worker_id}] 할당량: {len(my_docs)}개 (전체 {len(raw_data)}개 중)", flush=True)

        # 3. 모델 & DB 로드 (MIG 적용됨)
        model = SentenceTransformer(processor.valves.EMBED_PATH, device='cuda')
        client = chromadb.HttpClient(host=processor.valves.CHROMA_IP, port=processor.valves.CHROMA_PORT)
        collection = client.get_collection(processor.valves.CHROMA_COLLECTION_NAME)

        batch_size = processor.valves.EMBED_BATCH_SIZE
        batch = []
        processed = 0
        my_doc_count = 0

        # 4. 배치 처리
        batch_size = processor.valves.EMBED_BATCH_SIZE
        total = len(my_docs)
       
        for i in range(0, total, batch_size):
            batch = my_docs[i : i + batch_size]
            texts = [d['page_content'] for d in batch]
            metas = [d['metadata'] for d in batch]
            ids = [f"{uuid.uuid4()}_{k}_{worker_id}" for k in range(len(batch))]
            
            embeddings = model.encode(texts, batch_size=processor.valves.EMBED_BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True).tolist()
            collection.upsert(documents=texts, embeddings=embeddings, ids=ids, metadatas=metas)
           
            del texts, metas, ids, embeddings, batch
            if i % (batch_size * 5) == 0:
                gc.collect()
                torch.cuda.empty_cache()
           
            print(f"▶ [Worker-{worker_id}] 진행: {min(i+batch_size, total)}/{total}", flush=True)
           
        print(f"▶ [Worker-{worker_id}] ✅ 완료!", flush=True)

    except Exception as e:
        import traceback
        print(f"❌ [Worker-{worker_id}] 에러: {e}\\n{traceback.format_exc()}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    mode = os.environ.get("RUN_MODE", "FETCH")
    data_path = os.environ.get("DATA_PATH", "temp_docs.json")

    exit_code = 0

    try:
        if mode == "FETCH":
            run_fetch_mode(data_path)
        elif mode == "EMBED":
            w_id = int(os.environ.get("WORKER_ID", 0))
            t_w = int(os.environ.get("TOTAL_WORKERS", 1))
            run_embed_mode(data_path, w_id, t_w)
    except Exception as e:
        import traceback
        print(f"[FATAL] {e}\\n{traceback.format_exc()}", flush=True)
        exit_code = 1
    finally:
        import os
        os._exit(exit_code)
'''

            # 스크립트 파일 생성
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
                f.write(worker_script)
                temp_script_path = f.name
            temp_files.append(temp_script_path)

            data_file = tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            )
            data_file_path = data_file.name
            data_file.close()
            temp_files.append(data_file_path)

            base_env = os.environ.copy()
            for k, v in self.valves.dict().items():
                base_env[k] = str(v)
            base_env["DATA_PATH"] = data_file_path

            # ==================================================================
            # [Step 3] Phase 1: 데이터 수집 (Non-blocking 방식)
            # ==================================================================
            yield "\n📦 [Phase 1] 데이터 수집 (CPU)...\n"

            fetch_env = base_env.copy()
            fetch_env["RUN_MODE"] = "FETCH"

            p_fetch = subprocess.Popen(
                [sys.executable, temp_script_path],
                env=fetch_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            all_processes.append(p_fetch)

            # Non-blocking 설정
            try:
                os.set_blocking(p_fetch.stdout.fileno(), False)
            except Exception:
                pass

            fetch_buffer = ""
            while p_fetch.poll() is None:
                ready, _, _ = select.select([p_fetch.stdout], [], [], 0.1)
                if ready:
                    fetch_buffer, lines = self._read_process_output(
                        p_fetch, fetch_buffer
                    )
                    for line in lines:
                        yield line
                else:
                    time.sleep(0.01)

            # 남은 출력 처리
            try:
                remaining = p_fetch.stdout.read()
                if remaining:
                    yield fetch_buffer + remaining
                elif fetch_buffer:
                    yield fetch_buffer
            except Exception:
                if fetch_buffer:
                    yield fetch_buffer

            p_fetch.wait()

            if p_fetch.returncode != 0:
                yield f"🚨 수집 실패 (exit code: {p_fetch.returncode}). 파이프라인을 중단합니다.\n"
                return

            # 데이터 파일 크기 체크
            try:
                file_size_mb = os.path.getsize(data_file_path) / (1024 * 1024)
                yield f"📄 수집된 데이터 파일 크기: {file_size_mb:.2f} MB\n"
            except Exception:
                pass

            # ==================================================================
            # [Step 3.5] ChromaDB 초기화
            # ==================================================================
            try:
                import chromadb

                client = chromadb.HttpClient(
                    host=self.valves.CHROMA_IP, port=self.valves.CHROMA_PORT
                )
                try:
                    client.delete_collection(self.valves.CHROMA_COLLECTION_NAME)
                    yield "🧹 기존 ChromaDB 컬렉션 삭제\n"
                except Exception:
                    pass
                client.get_or_create_collection(self.valves.CHROMA_COLLECTION_NAME)
                yield "✅ ChromaDB 컬렉션 초기화 완료\n"
            except Exception as e:
                yield f"⚠️ ChromaDB 초기화 실패: {e}\n"
                return

            # ==================================================================
            # [Step 4] Phase 2: 병렬 임베딩 (안전 실행)
            # ==================================================================
            yield f"\n⚡ [Phase 2] {gpu_count}개 워커 순차 실행 시작...\n"

            base_env["RUN_MODE"] = "EMBED"
            base_env["TOTAL_WORKERS"] = str(gpu_count)

            for i, mig_uuid in enumerate(target_migs):
                embed_env = base_env.copy()
                embed_env["CUDA_VISIBLE_DEVICES"] = mig_uuid
                embed_env["WORKER_ID"] = str(i)

                # 시간차 실행 (Staggered Start) - Non-blocking 방식
                if i > 0:
                    yield f"⏳ 워커 {i} 시작 대기 중...\n"
                    for msg in self._wait_with_yield(
                        self.valves.WORKER_START_DELAY, f"   워커 {i} 대기"
                    ):
                        yield msg

                p = subprocess.Popen(
                    [sys.executable, temp_script_path],
                    env=embed_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                # Non-blocking 설정
                try:
                    os.set_blocking(p.stdout.fileno(), False)
                except Exception:
                    pass

                all_processes.append(p)
                active_processes.append(p)
                process_buffers[id(p)] = ""

                yield f"🚀 워커 {i} 실행됨 (PID: {p.pid}, MIG: {mig_uuid[:20]}...)\n"

            # ==================================================================
            # [Step 5] 로그 감시 (Select 방식)
            # ==================================================================
            yield "\n👀 로그 모니터링 시작...\n"
            start_time = time.time()
            last_activity = time.time()
            MAX_IDLE_SECONDS = 60.0
            MAX_TOTAL_SECONDS = 1200.0

            while active_processes:
                now = time.time()

                if now - start_time > MAX_TOTAL_SECONDS:
                    yield f"전체 실행 시간이 {MAX_TOTAL_SECONDS}초를 초과했습니다. 모든 워커를 강제 종료합니다.\n"
                    for p in active_processes:
                        try:
                            p.terminate()
                        except Exception:
                            pass
                    break

                error_detected = False
                # 종료된 프로세스 처리
                for p in active_processes[:]:
                    if p.poll() is not None:
                        # 남은 출력 모두 읽기
                        try:
                            remaining = p.stdout.read()
                            buf = process_buffers.get(id(p), "")
                            if remaining or buf:
                                yield buf + (remaining or "")
                        except Exception:
                            buf = process_buffers.get(id(p), "")
                            if buf:
                                yield buf

                        # 버퍼 정리
                        if id(p) in process_buffers:
                            del process_buffers[id(p)]

                        active_processes.remove(p)

                        status_ok = p.returncode == 0
                        # 종료 코드 로깅
                        status = (
                            "✅ 성공"
                            if status_ok
                            else f"❌ 실패 (code: {p.returncode})"
                        )
                        yield f"📋 워커 PID {p.pid} 종료: {status}\n"

                        if not status_ok:
                            error_detected = True

                if error_detected and active_processes:
                    yield "워커 에러 감지 -> 나머지 워커를 모두 강제 종료합니다\n"
                    for p in active_processes:
                        try:
                            p.terminate()
                        except Exception:
                            pass
                    break

                if not active_processes:
                    break

                # Select로 읽기 가능한 파이프 감지
                readable_pipes = [p.stdout for p in active_processes]
                try:
                    ready, _, _ = select.select(readable_pipes, [], [], 0.1)
                except (ValueError, OSError):
                    # 파이프가 이미 닫힌 경우
                    time.sleep(0.01)
                    continue

                if ready:
                    for pipe in ready:
                        # 해당 파이프의 프로세스 찾기
                        proc = next(
                            (p for p in active_processes if p.stdout == pipe), None
                        )
                        if not proc:
                            continue

                        buf_id = id(proc)
                        current_buffer = process_buffers.get(buf_id, "")
                        new_buffer, lines = self._read_process_output(
                            proc, current_buffer
                        )
                        process_buffers[buf_id] = new_buffer

                        if lines:
                            last_activity = time.time()

                        for line in lines:
                            yield line
                else:
                    time.sleep(0.01)

                if time.time() - last_activity > MAX_IDLE_SECONDS:
                    yield f"{MAX_IDLE_SECONDS}초 동안 워커 로그가 없어 모든 워커를 강제 종료합니다\n"
                    for p in active_processes:
                        try:
                            p.terminate()
                        except Exception:
                            pass
                    break

            # ==================================================================
            # [Step 6] 최종 결과 집계
            # ==================================================================
            yield "\n" + "=" * 50 + "\n"
            yield "📊 [최종 결과]\n"

            # Phase 1 (Fetch) 프로세스 제외하고 워커들만 체크
            embed_workers = [p for p in all_processes if p != p_fetch]

            failed_workers = []
            success_workers = []

            for i, p in enumerate(embed_workers):
                if p.returncode == 0:
                    success_workers.append(i)
                else:
                    failed_workers.append((i, p.pid, p.returncode))

            if failed_workers:
                yield f"❌ 실패한 워커: {len(failed_workers)}개\n"
                for worker_id, pid, code in failed_workers:
                    yield f"   - Worker-{worker_id} (PID: {pid}): exit code {code}\n"
                yield f"✅ 성공한 워커: {len(success_workers)}개\n"
                yield "\n⚠️ 일부 워커가 실패했습니다. 데이터가 불완전할 수 있습니다.\n"
            else:
                yield f"✅ 모든 워커 성공: {len(success_workers)}개\n"
                yield "\n✨ 모든 작업이 성공적으로 완료되었습니다!\n"

        except Exception as e:
            import traceback

            yield f"\n🚨 치명적 오류: {e}\n{traceback.format_exc()}\n"

        finally:
            # ==================================================================
            # [Cleanup] 리소스 정리
            # ==================================================================

            # 좀비 프로세스 방지
            for p in all_processes:
                if p.poll() is None:
                    try:
                        p.terminate()
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                    except Exception:
                        pass

            # 임시 파일 정리
            for f in temp_files:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception:
                    pass

            # 메모리 정리
            gc.collect()
