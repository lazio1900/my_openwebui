############################################
# 작성자 : 이민재
# 생성일자 : 2025-12-04
# 이력
# 2025-12-04 : 최초 생성 
#            : 데이터 처리 별도 구별
# 2025-12-24 : 전략금융 추가
# 2026-02-20 : 벡터db 실행시, 연관된 챗봇 pipe 재기동
############################################

##### 패키지 호출
import pandas as pd
from pydantic import BaseModel, Field
from typing import Union, Generator, Iterator
from typing import List, Dict, Literal
from typing import Callable, AsyncGenerator, Awaitable, Optional, Protocol
import base64
import io
import os
import re
import time
import logging
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import requests
import sys
from PIL import Image as PILImage
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import SentenceTransformerEmbeddings
from openpyxl import load_workbook
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
import torch
from openai import OpenAI
import json
from langchain_text_splitters import ExperimentalMarkdownSyntaxTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import random

from langchain.docstore.document import Document
from urllib.parse import urlparse, parse_qs

import math
import uuid

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage, AIMessage

# from open_webui.models.users import Users  # type: ignore
# from open_webui.utils.chat import generate_chat_completion  # type: ignore
# from open_webui.utils.misc import get_last_user_message  # type: ignore
from fastapi import Request
from fastapi.responses import StreamingResponse

from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage

from markdown import markdown

##### 파이프라인 개발
class Pipe:
    class Valves(BaseModel):
        DOC_PATH: str = "/data1/pipeline_data/auto_oper/"
        BASE_IMG_URL: str = "https://ai.wooricap.com/static/auto_oper_images/"
        STATIC_IMG_URL: str = (
            "/home/ubuntu/anaconda3/envs/webui/lib/python3.11/site-packages/open_webui/static/auto_oper_images/"
        )
        EMBED_PATH: str = "/home/ubuntu/pipelines/pipelines/embed_model/koe5"
        RERANK_PATH: str = "/home/ubuntu/pipelines/pipelines/rerank_model"
        CHROMA_PORT: int = 8800
        CHROMA_IP: str = "172.18.237.81"
        FAQ_COLLECTION_NAME: str = "faq_auto_oper"
        RULE_COLLECTION_NAME: str = "rule_auto_oper"
        STANDARD_COLLECTION_NAME_IMAGE: str = "auto_oper_standard_image"
        STANDARD_COLLECTION_NAME_TEXT: str = "auto_oper_standard_text"
        OLLAMA_URL: str = "http://localhost:11434"
        LLM_MODEL_NAME: str = "gpt-oss-120b"
        # DOC_NAME: str = "오토운영팀_FAQ_20250421.xlsx"

        OUTLINE_API_KEY: str = "ol_api_da8xs5ETkKt4qZ4ghC415TQ1B1U8R5idOh0xfK"
        OUTLINE_URL: str = "https://util.wooricap.com:7443"
        # COLLECTION_ID: str = "4efc49c2-f8c0-4f4e-b0f2-b5b926864acd"
        # RULE_COLLECTION_ID: str = "c6472f73-b9af-48a3-945e-8a16320f4ef6"
        STANDARD_COLLECTION_ID: str = "bfce3309-60d0-4477-a5cd-e8bda5f5e7c2"

    def __init__(self):
        self.valves = self.Valves()
        self.compression_retriever = None
        self.compressor = None
        self.rerank_model = None
        self.ensemble_retriever = None
        self.kiwi_retriever = None
        self.bm25_retriever = None
        self.rule_retriever = None
        self.faq_retriever = None
        self.stand_retriever = None
        self.pages = None
        self.embedding_function = None
        self.chroma_client = None
        self.device = None
        self.user = None
        self.request = None
        self.device = "cpu"
        self.event_emitter = None
        self.start_time = None
        self.memory = None
        self.oper_image_list = None
        self.image_retriever = None
        self.stand_loan_retriever = None
        self.stand_ems_retriever = None
        self.stand_save_retriever = None
        self.stand_dual_retriever = None
        self.stand_etc_retriever = None
        
        self.last_classified = ""

        self._need_restart_chatbot = True

        self.CHATBOT_FUNCTION_IDS = [
            "auto_oper_standard_used_car",
            "auto_oper_standard_stg_fin",
            "auto_oper_standard_used_lease",
            "auto_oper_standard_mid_truck",
            "auto_oper_agent",  # 통합 멀티에이전트 Pipe
        ]

 

    def _invalidate_chatbot_cache(self):
        """
        __init__ 시점에 app.state에 직접 접근하여
        챗봇 함수 4개의 캐시를 즉시 무효화합니다.
        UI에서 Save 누르는 순간 실행됩니다.
        """
        try:
            from open_webui.main import app

            for cid in self.CHATBOT_FUNCTION_IDS:
                if hasattr(app.state, "FUNCTIONS") and cid in app.state.FUNCTIONS:
                    del app.state.FUNCTIONS[cid]
                if hasattr(app.state, "FUNCTION_CONTENTS") and cid in app.state.FUNCTION_CONTENTS:
                    del app.state.FUNCTION_CONTENTS[cid]
                print(f"[vector_db] '{cid}' 캐시 무효화 완료")

            print(f"[vector_db] 챗봇 함수 {len(self.CHATBOT_FUNCTION_IDS)}개 무효화 완료")
        except Exception as e:
            print(f"[vector_db] 캐시 무효화 실패 (무시 가능): {e}")


    def pipe(
            self, 
            body: dict,
            __request__ = None
            ) -> Union[str, Generator, Iterator]:
        try:
            start_time = time.time()

            


            self.chroma_client = chromadb.HttpClient(
                host=self.valves.CHROMA_IP, port=self.valves.CHROMA_PORT
            )
            yield f"\n✅ chroma_client 객체 생성 \n"
            
            self.embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.valves.EMBED_PATH,
                    device=self.device,
                )
            )
            yield f"\n✅ embedding_function 객체 생성 \n"

            # wiki faq 데이터 불러오기
            documents = self.proc_wiki_faq(self.valves.STATIC_IMG_URL, self.valves.STANDARD_COLLECTION_ID)
            
            text_docs = []
            image_docs = []

            # 이미지와 텍스트 documents 구분하기
            for doc in documents:
                if  "#text" in doc.metadata['질문']:
                    d = self.text_doc_preprocesser(doc.metadata['답변'])
                    text_docs.append(d)
                elif "#image" in doc.metadata['질문']:
                    image_docs.append(doc)

            yield f"\n✅ text document 갯수 : {len(text_docs)}  \n"         
            yield f"\n✅ image document 갯수 : {len(image_docs)}  \n"         
            
            # 다중 list > 단일 list 형태로
            text_docs = list(self.flatten(text_docs))


            # 이미지 docs 저장
            self.save_to_chroma(image_docs, self.valves.STANDARD_COLLECTION_NAME_IMAGE)

            # 텍스트 docs 저장
            self.save_to_chroma(text_docs, self.valves.STANDARD_COLLECTION_NAME_TEXT)

            end_time = time.time()
            exe_time = end_time - start_time

            yield f"\n✅ Wiki 청킹 완료({exe_time:.2f})초\n"

            self._invalidate_chatbot_cache()           

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            yield f"exception :: {e}\n라인: {exc_tb.tb_lineno}"


        # finally:
        #     del documents
        #     if hasattr(self, "model"):
        #         del self.model
        #         self.model = None
        #         yield f"\n✅ model 정리 완료\n"

        #     with torch.cuda.device(3):
        #         torch.cuda.empty_cache()
        #     gc.collect()
        #     yield f"\n✅ 캐시 정리 완료\n"


    # ===================================================================
    # 함수 정의
    # ===================================================================

    # chroma 저장
    def save_to_chroma(self, documents, CHROMA_COLLECTION_NAME):
        try:
            try:
                self.chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
            except Exception as e:
                print(e)

            collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                embedding_function=self.embedding_function,
            )

            collection.upsert(
                documents=[document.page_content for document in documents],
                ids=[f"{CHROMA_COLLECTION_NAME}{i}" for i in range(len(documents))],
                metadatas=[document.metadata for document in documents],
            )

            if hasattr(self.chroma_client, "persist"):
                self.chroma_client.persist()
        finally:
            transport = getattr(self.chroma_client, "_transport", None)
            if transport and hasattr(transport, "close"):
                transport.close()

    # retrieve 함수화
    def retrieve(self, query):
        # 압축된 문서 검색
        compressed_docs = self.chroma_retriever.invoke(query)
        return compressed_docs

    def stream_output(self, input_string, delay):
        stream = io.StringIO(input_string)
        while True:
            line = stream.readline()
            if not line:
                break

            if line.strip().startswith("![Image]"):
                # 이미지 태그는 지연 없이 한 번에 출력
                yield line
            else:
                # 일반 텍스트는 한 글자씩 지연 출력
                for char in line:
                    yield char
                    time.sleep(delay)

    # ===================================================================
    # 데이터 전처리 함수
    # ===================================================================

    def proc_wiki_faq(self, STATIC_IMG_URL, COLLECTION_ID):
        # 이미지 저장 폴더 생성
        os.makedirs(self.valves.STATIC_IMG_URL, exist_ok=True)
        # 새로 수행시 전체 제거
        self.delete_all_files_in_directory(self.valves.STATIC_IMG_URL)

        # collection에서 documents 호출
        response_data = self.list_docs(COLLECTION_ID)

        # 전체 Document에 대해서 수행
        documents = []
        for data in response_data:
            # FAQ에서 질문 id 추출 (title)
            doc_id = data["id"]

            # FAQ에서 질문 추출 (title)
            title = data["title"]

            # FAQ에서 답변 추출 (text)
            text = data["text"]

            # url 추출
            source = self.valves.OUTLINE_URL + data["url"]

            ### 답변에서 이미지 추출 및 저장 (to .png)
            # 먼저 image urls 추출
            urls = self.extract_image_urls(text)
            # print("urls 가 어떤 output인지 확인 : ", urls)

            html_text = markdown("**질문** : \n" + title + "\n\n" + text)

            # 한 답변에 이미지가 2개 이상인 경우가 있음
            modified_text = text  # modified_text를 text로 초기화
            for idx, url in enumerate(urls):
                image_id = self.extract_id(url)
                file_name = self.download_image(url, doc_id, image_id)

                # 이미지 url을 태그 처리
                # 텍스트에서 링크를 <ImageInWiki> 태그로 변경
                modified_text = self.replace_image_url_to_tag(modified_text, file_name)

                # # html 형태 이미지 태그로 변경
                # modified_text = self.process_content_with_images(modified_text)

                # # markdown 결과를 html로 변경
                # html_text = markdown("**질문** : \n" + title + "\n\n" + modified_text)

            document = Document(
                page_content=html_text.strip().replace("\n", ""),
                metadata={
                    "title": title.strip().replace("\n", ""),
                    "질문": title.strip().replace("\n", ""),
                    "답변": modified_text,
                    "source": source,
                    "html_images" : self.process_content_with_images(modified_text),
                },
            )

            documents.append(document)

        return documents

    def convert_image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
                return f"![Image](data:image/png;base64,{encoded_string})"
        except FileNotFoundError:
            return "(이미지 파일을 찾을 수 없음)"

    def convert_image_to_url(self, image_path, image_name):
        if os.path.exists(image_path):
            # 이미지 URL 생성
            # return f'<img src="{self.valves.BASE_IMG_URL}{image_name}" alt="{image_name}" width="600" height="300">'
            # return f'<div class="image-container"><img src="{self.valves.BASE_IMG_URL}{image_name}" alt="{image_name}" class="fill-image"></div>'
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

    def ollama_request(self, question_user, model_name):
        headers = {"Content-Type": "application/json"}
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": question_user}],
            "stream": False,
            "options": {"temperature": 0},
        }

        try:
            r = requests.post(
                url=f"{self.valves.OLLAMA_URL}/api/chat",
                headers=headers,
                json=body,
            )

            r.raise_for_status()

            res_msg = r.json().get("message", {}).get("content", "")
            return res_msg

        except Exception as e:

            return f"Exception pipe ollama_request : {e}"

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
            print(e)

    def wiki_search(self, query: str, top_k: int) -> List[Document]:
        limit = 100
        json = {"query": query, "limit": limit, "statusFilter": ["published"]}

        raw_result = requests.post(
            f"{self.valves.OUTLINE_URL}/api/documents.search",
            json=json,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.valves.OUTLINE_API_KEY}",
            },
        )

        if not raw_result.ok:
            raise ValueError("Outline API returned an error: ", raw_result.text)

        results = raw_result.json()["data"]
        docs = []
        for result in results[:limit]:
            # 분류용 내용없는 상위문서들은 제외
            if result.get("context"):
                # 웹규정집을 검색할때는, 체크 아이콘이 있는게 최신 문서
                if (
                    result["document"]["collectionId"]
                    == "c6472f73-b9af-48a3-945e-8a16320f4ef6"
                    and result["document"]["icon"] != "✔️"
                ):
                    pass
                else:
                    if doc := self.result_to_document(result, 400):
                        docs.append(doc)
                        if len(docs) >= top_k:
                            break
        return docs

    def strip_tag(self, text) -> str:
        return re.sub(r"(\n|<b>|</b>)", "", text)

    def result_to_document(self, outline_res: dict, extra_chars: int) -> Document:
        text = outline_res["document"]["text"]
        context = outline_res["context"]

        clean_text = self.strip_tag(text)
        clean_context = self.strip_tag(context)
        index = clean_text.find(clean_context)

        if index == -1:
            expand_context = clean_context
        else:
            start = max(index - extra_chars, 0)
            end = min(index + len(clean_context) + extra_chars, len(text))
            expand_context = clean_text[start:end]

        metadata = {
            "id": outline_res["document"]["id"],
            "title": outline_res["document"]["title"],
            "source": self.valves.OUTLINE_URL + outline_res["document"]["url"],
            "text": text,
            "context": context,
            "ranking": outline_res["ranking"],
            "collection_id": outline_res["document"]["collectionId"],
            "parent_document_id": outline_res["document"]["parentDocumentId"],
            "revision": outline_res["document"]["revision"],
            "created_by": outline_res["document"]["createdBy"]["name"],
        }

        doc = Document(
            page_content=expand_context,
            metadata={**metadata},
        )
        return doc

    def request_completion_no_stream(
        self, prompt: str, model: str, client: OpenAI
    ) -> str:
        response = client.completions.create(model=model, prompt=prompt, temperature=0)
        return response.choices[0].text.strip()

    def clean_json_markdown(self, text):
        try:
            # ```json 또는 ```로 감싸진 부분 제거
            cleaned_text = re.sub(
                r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL
            )
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            print(cleaned_text)
            raise ValueError("유효한 JSON 형식이 아닙니다. 입력 데이터를 확인하세요.")

    def wiki_retrieve(self, chunk: List[Document], query: str) -> EnsembleRetriever:
        bm25_retriever = BM25Retriever.from_documents(chunk)
        bm25_retriever.k = 3

        embedding = SentenceTransformerEmbeddings(
            model_name=self.valves.EMBED_PATH, model_kwargs={"device": self.device}
        )

        vector = Chroma.from_documents(chunk, embedding)
        chroma_retriever = vector.as_retriever(k=3)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.3, 0.7],
        )

        rerank_model = HuggingFaceCrossEncoder(
            model_name=self.valves.RERANK_PATH, model_kwargs={"device": self.device}
        )
        compressor = CrossEncoderReranker(model=rerank_model, top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )

        return compression_retriever.invoke(query)

    def chunk_document(
        self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100, metadata=dict
    ) -> List[Document]:
        md_splitter = ExperimentalMarkdownSyntaxTextSplitter(
            return_each_line=False, strip_headers=False
        )
        structured_chunks = md_splitter.split_text(text)
        for chunk in structured_chunks:
            chunk.metadata = metadata

        return structured_chunks

    # ======================================================================
    # outline api 관련
    # ======================================================================
    # -----------------------------------------
    # 0. 안전한 요청: 429 → Retry-After 대기 후 재시도
    # -----------------------------------------
    def safe_post(self, url: str, json: dict, max_retry: int = 8):
        for attempt in range(max_retry):
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.valves.OUTLINE_API_KEY}",
            }

            r = requests.post(url, json=json, headers=headers, timeout=30)
            if r.status_code != 429:
                r.raise_for_status()
                print("auto_oper request 확인 : ", r.text)
                return r
            wait = int(r.headers.get("Retry-After", "10"))
            jitter = random.uniform(0, 3)
            sleep_sec = wait + jitter if wait else 10 + attempt * 2
            print(f"[RateLimit] {url}  →  재시도까지 {sleep_sec:.1f}s 대기")
            time.sleep(sleep_sec)
        raise RuntimeError(f"Too many retries for {url}")

    def list_docs(
        self, collection_id: str | None = None, limit: int = 100
    ) -> List[dict]:
        data = {"limit": limit} | (
            {"collectionId": collection_id} if collection_id else {}
        )
        docs, off = [], 0
        while True:
            r = self.safe_post(
                f"{self.valves.OUTLINE_URL}/api/documents.list", data | {"offset": off}
            )
            batch = r.json()["data"]
            if not batch:
                break
            docs.extend(batch)
            off += len(batch)
        return docs

    def load_documents(self, response_data):
        documents = []

        for item in response_data:
            document_id = item["id"]
            title = item["title"]
            text = item["text"]
            page_content = f"질문 : {title} \n 답변 : {text}"

            document = Document(
                page_content=page_content,
                metadata={
                    "document_id": document_id,
                    "질문": title,
                    "답변": text,
                },
            )
            documents.append(document)

        return documents

    def setup_storage(self):
        """이미지 저장 디렉토리 생성"""
        if not os.path.exists(self.valves.STATIC_IMG_URL):
            os.makedirs(self.valves.STATIC_IMG_URL)

    def extract_image_urls(self, markdown_text):
        """
        Markdown 텍스트에서 이미지 URL 추출하는 함수
        Args:
            markdown_text (str): 대상 문자열
        Returns:
            list: 추출된 URL 리스트 (URL이 없으면 빈 리스트 반환)
        """
        pattern = r"!\[\]\((.*?)\)"
        matches = re.findall(pattern, markdown_text)
        urls = []
        for match in matches:
            url = match.split()[0]
            urls.append(self.valves.OUTLINE_URL + url)
        return urls

    def extract_id(self, url):
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        return query_params.get("id", [None])[0]

    def download_image(self, url, document_id, image_id):
        """이미지 다운로드 및 문서 연관 저장"""
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
                print(f"Failed to download {url}: Status Code {response.status_code}")
                return filename
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return None
        

    def replace_image_url_to_tag(self, text, file_name):
        """
        텍스트 내의 이미지 링크를 <ImageInWiki> 태그로 변경합니다.

        Args:
            text: 원본 텍스트.
            document_id: 문서 ID.

        Returns:
            이미지 링크가 변경된 텍스트.
        """

        image_link_pattern = r"!\[\]\((.*?)\)"
        match = re.search(image_link_pattern, text)

        if match:
            replacement = f"<ImageInWiki>{file_name}</ImageInWiki>"
            return text.replace(match.group(0), replacement)
        else:
            return text

    # ======================================================================
    # outline 웹규정집 관련
    # ======================================================================

    def proc_wiki_rule(self, collectionId, filter_docs):
        docs = []

        response_data = self.list_docs(collectionId)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.OUTLINE_API_KEY}",
        }

        for result in response_data:
            # 웹규정집은 체크 표시가 있는 문서만 유효
            if collectionId == "c6472f73-b9af-48a3-945e-8a16320f4ef6":
                if result["icon"] != "✔️":
                    continue

            # Apply title filtering if filter_docs is provided
            if filter_docs:  # Only apply when filter_docs is not empty
                title_matches = any(
                    substring in result["title"] for substring in filter_docs
                )
                if not title_matches:
                    continue

            doc_id = result["id"]

            detail_result = requests.post(
                f"{self.valves.OUTLINE_URL}/api/documents.info",
                json={"id": doc_id},
                headers=headers,
            )
            dr = detail_result.json()["data"]

            metadata = {
                "id": dr["id"],
                "title": dr["title"],
                "source": self.valves.OUTLINE_URL + dr["url"],
                "text": dr["text"],
                "collection_id": dr["collectionId"],
                "parent_document_id": dr["parentDocumentId"],
                "revision": dr["revision"],
                "created_by": dr["createdBy"],
                "updated_by": dr["updatedBy"],
                "created_at": dr["createdAt"],
                "updated_at": dr["updatedAt"],
            }

            doc = Document(
                page_content="",
                metadata={**metadata},
            )
            docs.append(doc)
        return docs

    def make_structure(self, all_text):
        chapter_pattern = r"제\s*(\d+)\s*장\s*([^\n]*)"
        section_pattern = r"제\s*(\d+)\s*절\s*([^\n]*)"
        article_pattern = r"제\s*(\d+)\s*조\s*\(?([^)]*)?\)?\s*(.*?)(?=\n#+\s*제\s*\d+\s*장|\n제\s*\d+\s*절|\n제\s*\d+\s*조|\n<별표|\n<별지서식|\n##\s+부칙|\Z)"
        annexes_and_forms_pattern = r"##\s*<(별표|별지서식)\s*(\d*)>([\s\S]*?)(?=\n##\s*<(?:별표|별지서식)\s*\d*>|\n##\s+부칙|\Z)"

        structure = []
        current_chapter = None
        current_section = None

        # 순차적으로 텍스트 탐색
        for match in re.finditer(
            rf"{chapter_pattern}|{section_pattern}|{article_pattern}|{annexes_and_forms_pattern}",
            all_text,
            re.DOTALL,
        ):
            chapter, chapter_title = match.group(1, 2)
            # print(f"{i}번째 chapter, chapter_title : ", chapter, chapter_title)
            section, section_title = match.group(3, 4)
            # print(f"{i}번째 section, section_title : ", section, section_title)
            article_number, article_title, article_content = match.group(5, 6, 7)
            # print(f"{i}번째 article_number, article_title article_content : ", article_number, article_title, article_content, "\n")

            # 장 탐색
            if chapter:
                current_chapter = {
                    "chapter": f"제{chapter}장",
                    "title": chapter_title.strip(),
                    "sections": [],
                }
                structure.append(current_chapter)
                current_section = None  # 새로운 장이 시작되면 절 초기화

            # 절 탐색
            elif section:
                current_section = {
                    "section": f"제{section}절",
                    "title": section_title.strip(),
                    "articles": [],
                }
                if current_chapter:
                    current_chapter["sections"].append(current_section)

            # 조 탐색 및 조 내용 처리
            elif article_number:
                if current_chapter is None:
                    current_chapter = {
                        "chapter": "장 없음",
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
                    # 절이 없으면(또는 안 쓰이면) 장에 직접 추가
                    current_chapter["sections"].append(
                        {"section": None, "articles": [article]}
                    )
                # elif current_chapter:
                #     current_chapter["sections"].append({"section": None, "articles": [article]})

        # 별표 및 별지서식 탐색
        for match in re.finditer(annexes_and_forms_pattern, all_text, re.DOTALL):
            # item_type, number, content = match.group(1, 2, 3)
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

    def extract_tables_with_context(self, page):
        tables_with_context = []
        content_elements = []

        tables = page.extract_tables()
        if not tables:
            # 표가 없으면 전체 텍스트를 반환
            text = page.extract_text() or ""
            if text.strip():
                content_elements.append(
                    {
                        "type": "text",
                        "content": text.strip(),
                        "y_position": 0,  # 페이지 상단부터 시작
                    }
                )
            return content_elements

        page_width, page_height = page.width, page.height

        # 모든 표 추출
        for i, table in enumerate(tables):
            # 표의 위치 정보(bbox)를 이용하여 위치 추정
            table_bbox = page.find_tables()[i].bbox
            x0, y0, x1, y1 = table_bbox

            # 표 내용에서 None 값 및 줄바꿈 문자 제거
            table = [
                [(str(cell).strip() if cell is not None else "") for cell in row]
                for row in table
            ]

            # 표 정보를 위치 정보와 함께 저장
            tables_with_context.append(
                {
                    "type": "table",
                    "content": table,
                    "y_position": y0,
                    "bbox": table_bbox,
                }
            )

        # 표를 기준으로 텍스트를 추출하여 나눔
        last_y_position = 0
        for table in tables_with_context:
            y0 = table["y_position"]
            if last_y_position < y0:
                # 표 이전의 텍스트를 추출
                text_segment = (
                    page.crop((0, last_y_position, page_width, y0)).extract_text() or ""
                )
                if text_segment.strip():
                    content_elements.append(
                        {
                            "type": "text",
                            "content": text_segment.strip(),
                            "y_position": last_y_position,
                        }
                    )

            # 표 자체를 추가
            content_elements.append(table)

            # 다음 텍스트 추출을 위해 위치 갱신
            last_y_position = table["bbox"][3]

        # 마지막 표 이후의 텍스트 추가
        if last_y_position < page_height:
            below_text = (
                page.crop((0, last_y_position, page_width, page_height)).extract_text()
                or ""
            )
            if below_text.strip():
                content_elements.append(
                    {
                        "type": "text",
                        "content": below_text.strip(),
                        "y_position": last_y_position,
                    }
                )

        return content_elements

    def extract_only_text(self, page):
        content_elements = []

        # 표가 없으면 전체 텍스트를 반환
        text = page.extract_text() or ""
        if text.strip():
            content_elements.append(
                {
                    "type": "text",
                    "content": text.strip(),
                    "y_position": 0,  # 페이지 상단부터 시작
                }
            )
        return content_elements

    def table_to_html(self, table):
        # 표를 DataFrame으로 변환하고, 마크다운 형식으로 변환
        df = pd.DataFrame(table[1:], columns=table[0])
        html_table = df.to_html(index=False, escape=False)
        return html_table

    def insert_tables_into_text(self, text, tables_with_context):
        final_text = ""
        current_position = 0

        # 표들을 y 좌표 기준으로 정렬합니다.
        sorted_tables = sorted(tables_with_context, key=lambda x: x["bbox"][1])

        for item in sorted_tables:
            # HTML 형식으로 표 변환
            html_table = self.table_to_html(item["table"])

            # 표가 삽입될 위치에 따라 텍스트를 추가
            if current_position < len(text):
                final_text += text[current_position:].strip() + "\n"

            # 표 삽입
            final_text += html_table + "\n"

            # 현재 위치를 표 삽입 이후로 갱신
            current_position += len(html_table)

        return final_text

    def process_content_elements(self, content_elements):
        # 위치를 기준으로 정렬하여 전체 페이지를 처리
        sorted_elements = sorted(content_elements, key=lambda x: x["y_position"])
        final_text = ""

        for element in sorted_elements:
            if element["type"] == "text":
                final_text += element["content"] + "\n"
            elif element["type"] == "table":
                html_table = self.table_to_html(element["content"])
                final_text += html_table + "\n"

        return final_text

    def find_common_header(self, texts):
        # 모든 텍스트에서 공통으로 시작하는 문자열을 찾아서 반환
        if not texts:
            return ""
        min_length = min(len(text) for text in texts)
        common_prefix = ""
        for i in range(min_length):
            char_set = set(text[i] for text in texts)
            if len(char_set) == 1:
                common_prefix += char_set.pop()
            else:
                break
        return common_prefix.strip()

    def multiple_convert_to_documents(self, docs):
        all_documents = []

        for doc in docs:
            title = doc.metadata["title"]
            all_text = doc.metadata["text"]
            source = doc.metadata["source"]
            structure = self.make_structure(all_text)
            chapters, etc_sections = self.classify_sections(structure)
            chapters_docs = self.process_chapter_section(chapters, title, source)
            etc_sections_docs = self.process_etc(etc_sections, title, source)
            all_documents.extend(chapters_docs)
            all_documents.extend(etc_sections_docs)

        return all_documents

    def pretty_print_documents(self, documents):
        # 각 Document 객체의 내용을 보기 좋게 출력
        for idx, doc in enumerate(documents, start=1):
            print(f"Document {idx}:")
            print(f"Page Content:\n{doc.page_content}\n")
            print("Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
            print("-" * 80)

    # 장절조 process
    def process_chapter_section(self, structure, title, source):
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
                    article_content = "\n".join(article["content"])
                    # soup = BeautifulSoup(article_content, "html.parser")
                    # article_content_no_html = soup.get_text()
                    # article_content_no_html = re.sub(r'\n{2,}', '\n', article_content_no_html)
                    # article_content_no_html = re.sub(r'[ \t]+', ' ', article_content_no_html)

                    # page_content 형식 구성: [article_number] [title] [content]
                    page_content = f"[{title}] \n [{chapter_number}] [{chapter_title}] \n [{article_number}] [{article_title}] {article_content.strip()}"
                    # page_content_no_html = f"[{article_number}] [{article_title}] {article_content_no_html.strip()}"

                    # 메타데이터 구성
                    metadata = {
                        "chapter": chapter_number,
                        "chapter_title": chapter_title,
                        "section": section_number,
                        "section_title": section_title,
                        # "article_content": page_content,
                        "title": title,
                        "source": source,
                    }

                    # Document 객체 생성 및 추가
                    documents.append(
                        Document(page_content=page_content, metadata=metadata)
                    )
                    # 조 merge 결과 생성
                    chapter_merge.append(page_content)
            # 조 merge 결과를 documents에 append
            page_content_merge = "\n".join(chapter_merge)
            chapter_meta = {
                "chapter": chapter_number,
                "chapter_title": chapter_title,
                "title": title,
                "chpater_merge_yn": "Y",
            }
            documents.append(
                Document(page_content=page_content_merge, metadata=chapter_meta)
            )
            chapter_merge = []

        return documents

    # 별첨 process
    def process_etc(self, structure, title, source):
        documents = []
        for etc in structure:
            etc_content = etc.get("etc_content", "")
            # print("etc_title 입니다 : ", etc_title )
            etc_title = etc.get("etc_title", "")
            if etc_title:
                etc_content = "\n".join(etc_content)
                # soup = BeautifulSoup(etc_content, "html.parser")

                # etc_content_no_html = soup.get_text()
                # etc_content_no_html = re.sub(r'\n{2,}', '\n', etc_content_no_html)
                # etc_content_no_html = re.sub(r'[ \t]+', ' ', etc_content_no_html)

                metadata = {
                    "etc_title": etc_title,
                    # "etc_content": etc_content,
                    "title": title,
                    "source": source,
                }

                documents.append(
                    Document(
                        page_content=f"[{title}] \n [{etc_title}] \n {etc_content.strip()}",
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

    def extract_chapter_text(self, pages, chapter=None, etc_title=None, title=None):
        extracted_text = []
        for page in pages:
            if (
                chapter
                and "chapter" in page.metadata
                and page.metadata["chapter"] == chapter
                and page.metadata["title"] == title
            ) or (
                etc_title
                and "etc_title" in page.metadata
                and page.metadata["etc_title"] == etc_title
                and page.metadata["title"] == title
            ):

                extracted_text.append(f"{page.page_content} \n")

        return "".join(extracted_text)
    

    def extract_html_images_url(self, documents):
        oper_image_list = {}
        for doc in documents:           
            if "재고금융 운영기준 알려줘" in doc.metadata['질문']:
                oper_image_list['재고금융'] = doc.metadata['html_images']

            if "ESM, 딜러 전용 대출(딜러 대출) 상품 운영기준 알려줘" in doc.metadata['질문']:
                oper_image_list['임직원대출'] = doc.metadata['html_images']        

            if "중고리스 운영기준 알려줘" in doc.metadata['질문']:
                oper_image_list['중고리스'] = doc.metadata['html_images']        

            if "신용구제 상품 운영기준 알려줘" in doc.metadata['질문']:
                oper_image_list['신용구제'] = doc.metadata['html_images']        

            if "할부 운영기준 알려줘" in doc.metadata['질문']:
                oper_image_list['할부'] = doc.metadata['html_images']        

            if "론 운영기준 알려줘" in doc.metadata['질문']:
                oper_image_list['론'] = doc.metadata['html_images']          

            return oper_image_list                


    # 텍스트 분할
    def split_text_by_separator(self, text, separator="###"):   
        """
        텍스트를 지정된 구분자로 나누고, 빈 조각이나 단순 '\\' 조각은 제외한다.
        """
        chunks = []
        # 현재는 하나의 문자열만 받아서 리스트에 감싸지만,
        # 여러 문서를 한 번에 처리하고 싶다면 for doc in documents: 로 바꾸면 된다.
        for doc in [text]:
            for part in doc.split(separator):
                # 앞·뒤 공백(와 개행) 제거
                cleaned = part.strip()
                # 빈 문자열이 아니고, 오직 역슬래시 하나만 있는 경우도 제외
                if cleaned and cleaned != '\\':
                    chunks.append(cleaned)
        return chunks    

    def text_doc_preprocesser(self, sub_doc):
        document = []
        docs_list = self.split_text_by_separator(sub_doc)
        # print(docs_list)
        for doc in docs_list:
            pattern = r"\((.*?)\)"  # 괄호 안의 내용을 추출하는 정규 표현식
            subtask = re.search(pattern, doc)[0]

            d = Document(metadata = 
                            {
                                # "task":task,
                                "subtask":subtask,
                                },
                            page_content=doc)
            document.append(d)
        return document

    def flatten(self, seq):
        """다중 중첩 리스트를 모두 한 단계로 풀어 반환합니다."""
        for item in seq:
            if isinstance(item, (list, tuple)):   # 원하는 반복 가능한 타입 지정
                yield from self.flatten(item)           # 재귀 호출
            else:
                yield item

    ############################################################
    #   langgraphtic 관련 함수들
    ############################################################
    def faq_retrieve(self, question):
        """
        FAQ 문서를 검색합니다

        Args:
            state (dict): 현재 그래프의 상태

        Returns:
            state (dict): 검색된 문서를 포함한 새로운 상태
        """
        print("--FAQ 검색---")

        documents = self.faq_retriever.invoke(question)
        return documents

    def rule_retrieve(self, question):
        """
        RULE 문서를 검색합니다

        Args:
            state (dict): 현재 그래프의 상태

        Returns:
            state (dict): 검색된 문서를 포함한 새로운 상태
        """
        print("---규정 검색---")

        documents = self.rule_retriever.invoke(question)
        return documents
    
    def stand_retrieve(self, question):
        """
        운영기준 문서를 검색합니다

        Args:
            state (dict): 현재 그래프의 상태

        Returns:
            state (dict): 검색된 문서를 포함한 새로운 상태
        """
        print("---운영기준 검색---")

        documents = self.stand_retriever.invoke(question)
        return documents


    async def grade_documents(self, user_message, documents, FLAG="FAQ"):

        # 데이터 모델 정의
        class GradeDocuments(BaseModel):
            binary_score: str = Field(
                description="문서와 질문의 연관성 여부. (예 or 아니오)"
            )

        # 출력 파서 정의
        parser = PydanticOutputParser(pydantic_object=GradeDocuments)

        # 프롬프트 정의
        prompt = """
        당신은 사용자의 질문에 대해 검색된 문서의 관련성을 평가하는 전문가입니다.
        문서에 질문과 관련된 키워드나 의미가 담겨 있으면, 해당 문서를 ‘관련 있음’으로 평가하세요.
        질문과 문서과 관련이 있는지에 대해서 **매우 엄격**하게 판단하세요.
        다시 한번 강조합니다. 질문과 문서과 관련이 있는지에 대해서 **매우 엄격**하게 판단하세요.
        문서가 질문과 관련이 있는지 여부를 ‘예’ 또는 ‘아니오’로 표시해 주세요.
        다음 JSON 스키마를 정확히 준수하여 한국어로만 응답하시오.
        {{
        "binary_score": "예|아니오"
        }}

        질문: {question} 

        문맥: {context} 

        """

        print("---문서와 질문의 연관성 평가---")
        filtered_docs = []
        next_answer = "아니오"
        count_y = 0
        count_n = 0

        send_status = get_send_status(self.event_emitter)

        for idx, d in enumerate(documents):
            await send_status(
                status_message=f"{FLAG} 관련 문서 확인 중...({idx+1} / {len(documents)})",
                done=False,
            )
            try:
                replaced_prompt = prompt.format(question=user_message, context=d)
                data_json = {
                    "model": self.valves.LLM_MODEL_NAME,
                    "messages": [{"role": "user", "content": replaced_prompt}],
                    "stream": False,
                    "temperature": 0,
                }
                response = await generate_chat_completion(
                    self.request, data_json, self.user
                )
                content = response["choices"][0]["message"]["content"]

                score = parser.parse(content)

                if score.binary_score == "예":
                    print("---평가: 연관 문서---")
                    filtered_docs.append(d)
                    count_y += 1
                else:
                    print("---평가: 연관 없는 문서---")
                    count_n += 1

            except Exception as e:
                print(f"Error parsing response: {e}")

        if count_y == 0:
            next_answer = "예"

        print(f"관련 문서 개수: {count_y}, 관련 없는 문서 개수: {count_n}")
        return filtered_docs, next_answer

    async def common_generate(self, user_message, history):

        # 테스트 데이터 생성
        system = """
        지금 너에게 물어보는 경우에는 '론/할부', '임직원대출(ESM)', '신용구제', 'Dual Offer' 업무 분류에 벗어난 질문을 한 상태야
        재질문으로 사용자에게 다시 입력을 유도해.
        보통은 "내용에서 벗어난 질문입니다. 중고승용 업무 중 '론/할부', '임직원대출(ESM)', '신용구제', 'Dual Offer' 중 문의해주세요." 정도로 답하고
        가능하면 사용자 질문이나 그간 대화내용을 참고해서 더 나은 질문에 대해서 답변해줘. 
        예를들어 "운영기준 알려줘"와 같은 너무 넓은 범위 질문이 들어오면,
        추천질문으로 "론/할부 운영기준 알려줘", '임직원대출 운영기준 알려줘' 와 같은 추천질문을 제공함으로써, 사용자가 업무 영역으로 넘어가도록 추천질문을 제공해줘.
        너는 반드시 "한국어"로 답변해.               
        """

        # history = lambda _: self.memory.load_memory_variables({})["history"]

        # Define the prompt template
        common_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "대화내용 : {history}"),
                ("human", "사용자 질문: {question}"),
            ]
        )

        replaced_prompt = common_prompt.format(history=history, question=user_message)

        data_json = {
            "model": self.valves.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": replaced_prompt}],
            "stream": True,
            "temperature": 0,
        }
        send_status = get_send_status(self.event_emitter)
        end_time = time.time()
        exe_time = end_time - self.start_time
        await send_status(status_message=f"완료: {exe_time:.2f}초", done=True)

        # 채팅 완성 비동기 처리
        response = await generate_chat_completion(self.request, data_json, self.user)


        # 응답 내용 추출
        return response

    async def generate(self, documents, user_message, history, classified):

        # 테스트 데이터 생성
        system = """
        너는 JB우리캐피탈 오토운영팀 챗봇이야.
        너가 대답하는 경우는 오토운영팀 중고차 대출 상품에 대해 상세 문의 하는 경우야
        큰 업무 분류 기준은 '론/할부', '임직원대출(ESM)', '신용구제', 'Dual Offer', '공통' 이며, 
        사용자는 특정 업무가 분류된 이후로 업무에 관련된 상세 질문을 너에게 물어볼거야 (상품명, 대상, 금리, 프로모션, 판촉수수료, 슬라이딩, 연체이자율, 중도상환, 실적 등)
        사용자에게 주어진 질문에 대해서 참고자료를 활용해서 적절히 답변해
        <업무참고> 를 통해서 업무에 사용되는 약어 등을 참고해

        <업무참고>
        1. G/L : guideline 금리를 의미
        2. 금리등급 : 내부등급으로 전국민 등급인 Nice등급과 다르게 JB우리캐피탈에서 사용하는 내부 금리 등급
        3. 증빙 : 증빙서류 제출시 Nego가능
        4. 거점장 : 거점장 전결로 Nego 가능
        5. N1, N2, N3 등 : Nice 등급을 의미함. 예를들어 내부 등급 대상 중 Nice 상위등급자(N1, N2 등)에게는 Nego 제안 가능

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
        

        """

        # Define the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "대화내용 : {history}"),
                ("human", "참고자료 : {context}"),
                ("human", "사용자 질문: {question}"),
            ]
        )

        replaced_prompt = prompt.format(
            history=history, context=documents, question=user_message
        )

        data_json = {
            "model": self.valves.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": replaced_prompt}],
            "stream": True,
            "temperature": 0,
        }

        # 추론 멘트 생성
        send_status = get_send_status(self.event_emitter)
        end_time = time.time()
        exe_time = end_time - self.start_time
        await send_status(status_message=f"완료: {exe_time:.2f}초", done=True)


        # 채팅 완성 비동기 처리
        response = await generate_chat_completion(self.request, data_json, self.user)
        # self.memory.save_context(inputs = {"human": user_message}, outputs = {"ai": response})


        def check_image_bool(classified, last_classified):
            if last_classified == "":
                return True
            elif classified != last_classified:
                return True
            else:
                return False
        
        if check_image_bool(classified, self.last_classified):
            response = self.oper_image_list[classified] + f"{classified} 운영기준 입니다. 운영기준에 대해 궁금한 사항을 문의해주세요."
        else:
            # 참조 생성
            send_citation = get_send_citation(self.event_emitter)
            for idx, obj in enumerate(documents, start=1):
                await send_citation(
                    # url=f"출처{idx+1}", title=obj.metadata['title'], content=obj.page_content
                    
                    ### 출처1, 2, 3 방식
                    url=f"출처{idx}",
                    title=f"출처{idx}",
                    content=obj.page_content,

                    ### 출처 연결 방식
                    # url = obj.metadata['source'],
                    # title = f"출처{idx}",
                    # content = obj.page_content,
                )
            

        # 응답 내용 추출
        return response



    async def etc_generate(self, documents, user_message, history):

        # 테스트 데이터 생성
        system = """
        너는 JB우리캐피탈 오토운영팀 챗봇이야.
        너가 대답하는 경우는 오토운영팀 중고차 대출 상품에 대해 상세 문의 하는 경우야
        사용자에게 주어진 질문에 대해서 참고자료를 활용해서 적절히 답변해

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
                 
        """

        # history = lambda _: self.memory.load_memory_variables({})["history"]

        # Define the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "대화내용 : {history}"),
                ("human", "참고자료 : {context}"),
                ("human", "사용자 질문: {question}"),
            ]
        )

        replaced_prompt = prompt.format(
            history=history, context=documents, question=user_message
        )
        
        data_json = {
            "model": self.valves.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": replaced_prompt}],
            "stream": True,
            "temperature": 0,
        }
        send_status = get_send_status(self.event_emitter)
        end_time = time.time()
        exe_time = end_time - self.start_time
        await send_status(status_message=f"완료: {exe_time:.2f}초", done=True)

        # 채팅 완성 비동기 처리
        response = await generate_chat_completion(self.request, data_json, self.user)


        # 응답 내용 추출
        return response




    async def requery_and_classify_task(self, question):
        """
        문의 내용이 어느 업무인지 분류합니다.

        Args:
            question (str): 사용자 질문.
            

        Returns:
            classifed_task (str) : 분류된 업무
        """

        # 데이터 모델 정의
        class TotalTask(BaseModel):
            classifed_task: str = Field(
                description="사용자 질문에 대한 업무 분류. ('론/할부', '임직원대출(ESM)', '신용구제', 'Dual Offer', '재질의', '공통')"
            )


        # 출력 파서 정의
        parser = PydanticOutputParser(pydantic_object=TotalTask)


        # 프롬프트 정의
        prompt = """당신은 사용자의 질문에 대해 어떤 업무인지 분류하는 분류 모델입니다
        아래 업무 정의와 관련 질의를 통해, 입력된 문의가 어느 업무인지 판단하여 분류하시오.
        분류 기준은 '론/할부', '임직원대출(ESM)', '신용구제', 'Dual Offer', '공통', '재질의' 이며 5개 업무와 1개 업무를 벗어난 '재질의'로 분류하시오.
        사용자는 특정 업무가 분류된 이후로 업무에 관련된 상세 질문을 추가할 수 있습니다.(상품명, 대상, G/L금리, Nice 등급, Nego기준, 프로모션, 판촉수수료, 슬라이딩, 연체이자율, 중도상환, 실적 등) 이점 참고하세요
        
        <론/할부>
        정의 : 론/할부는 일반고객에 대해 중고차 대출을 전액대출 형태(론)와 할부납부하는 형태(할부)로 상품이 구분됨
        관련 질의 : 할부 운영기준에 대해서 알려줘, 론 운영기준에 대해서 알려줘

        <임직원대출(ESM)>
        정의 : 일반고객 대출 외 임직원에 대한 대출이며, 일반 임직원과 딜러 상품이 별도 존재함
        관련 질의 : 임직원대출 운영기준에 대해서 알려줘

        <신용구제>
        정의 : 신용회복한 고객(신용구제 대상)을 대상으로 중고차 대출해주는 상품
        관련 질의 : 신용구제 운영기준에 대해서 알려줘

        <Dual offer>
        정의 : 대출한도 높게 원하는 사람들한테 한도 늘려주고 금리 높게 해주는 상품
        관련 질의 : Dual offer 운영기준에 대해서 알려줘

        <공통>
        정의 : 공통 업무 영역
        관련 질의 : Nice 점수(등급)에 관해서 알려줘

        <재질의>
        정의 : 론/할부, 임직원대출(ESM), 신용구제, dual offer 외 상품 문의(업무관련 공통 질의 포함) 또는 상품 중 모호한 질의인 경우 재질의 실시
        관련 질의 : 전략금융 상품에 대해서 알려줘 또는 운영기준 알려줘
        예상 답변 : 론/할부, 임직원대출(ESM), 신용구제에 대해서 문의해주세요. 또는 론/할부, 임직원대출(ESM), 신용구제, Dual offer 중 어떤 업무에 관한 문의인가요?
            
        
        다음 JSON 스키마를 정확히 준수하여 한국어로만 응답하시오.
        {{
        "classifed_task": "론/할부|임직원대출(ESM)|신용구제|재질의|Dual Offer|공통"
        }}

        질문: {question} 
        

        """

        send_status = get_send_status(self.event_emitter)

        await send_status(status_message="질문 분류 중 ...", done = False)

        replaced_prompt = prompt.format(question=question)

        data_json = {
            "model": self.valves.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": replaced_prompt}],
            "stream": False,
            "temperature": 0,
        }

        response = await generate_chat_completion(self.request, data_json, self.user)
        content = response["choices"][0]["message"]["content"]
        response_json = parser.parse(content)
        res = response_json.classifed_task

        return res


