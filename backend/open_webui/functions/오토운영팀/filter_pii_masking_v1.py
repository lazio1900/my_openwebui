############################################
# PII 마스킹 필터 (개인정보 보호)
# 작성자 : 이민재 / Claude
# 생성일자 : 2026-03-18
# 이력
# 2026-03-18 v1 : 한국 PII 마스킹 필터 초기 버전
#   - Open WebUI Filter 타입 (inlet + outlet)
#   - 한국 주요 개인정보 정규식 기반 마스킹
#   - 지원 PII: 주민등록번호, 휴대폰/전화번호, 이메일,
#     신용카드번호, 계좌번호, 여권번호, 운전면허번호,
#     사업자등록번호, IP주소
#   - Valve로 PII 유형별 ON/OFF 제어
#   - inlet: 사용자 입력 마스킹 (LLM에 전달 전)
#   - outlet: LLM 응답 마스킹 (사용자에게 전달 전)
############################################

import re
import logging
from typing import Optional
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class Filter:
    """
    한국 PII 마스킹 필터

    Open WebUI Filter 타입으로 동작:
    - inlet: 사용자 메시지에서 개인정보를 마스킹한 후 LLM에 전달
    - outlet: LLM 응답에서 개인정보를 마스킹한 후 사용자에게 전달

    관리자 페이지 → Functions에서 이 필터를 등록하고,
    is_global=True로 설정하면 모든 모델에 자동 적용됩니다.
    """

    class Valves(BaseModel):
        """관리자 설정 (Admin Panel → Functions → Valves)"""
        priority: int = Field(
            default=0,
            description="필터 실행 우선순위 (낮을수록 먼저 실행)"
        )
        # ── PII 유형별 ON/OFF ──
        MASK_RRN: bool = Field(
            default=True,
            description="주민등록번호 마스킹 (900101-1234567)"
        )
        MASK_PHONE: bool = Field(
            default=True,
            description="휴대폰/전화번호 마스킹 (010-1234-5678)"
        )
        MASK_EMAIL: bool = Field(
            default=True,
            description="이메일 주소 마스킹 (user@example.com)"
        )
        MASK_CARD: bool = Field(
            default=True,
            description="신용카드번호 마스킹 (1234-5678-9012-3456)"
        )
        MASK_ACCOUNT: bool = Field(
            default=True,
            description="은행 계좌번호 마스킹 (110-123-456789)"
        )
        MASK_PASSPORT: bool = Field(
            default=True,
            description="여권번호 마스킹 (M12345678)"
        )
        MASK_DRIVER_LICENSE: bool = Field(
            default=True,
            description="운전면허번호 마스킹 (11-12-123456-78)"
        )
        MASK_BRN: bool = Field(
            default=True,
            description="사업자등록번호 마스킹 (123-45-67890)"
        )
        MASK_IP: bool = Field(
            default=False,
            description="IP 주소 마스킹 (192.168.1.1) - 기본 OFF"
        )
        # ── 마스킹 동작 설정 ──
        ENABLE_INLET: bool = Field(
            default=True,
            description="inlet(사용자 입력) 마스킹 활성화"
        )
        ENABLE_OUTLET: bool = Field(
            default=True,
            description="outlet(LLM 응답) 마스킹 활성화"
        )
        LOG_MASKED: bool = Field(
            default=True,
            description="마스킹 발생 시 로그 기록 (마스킹된 내용은 로그에 포함 안 됨)"
        )

    def __init__(self):
        self.valves = self.Valves()

    # ──────────────────────────────────────────
    # PII 정규식 패턴 정의
    # ──────────────────────────────────────────

    @staticmethod
    def _get_patterns(valves) -> list[tuple[str, re.Pattern, str]]:
        """
        활성화된 PII 유형에 대한 (이름, 패턴, 대체문자열) 리스트 반환.
        패턴 순서가 중요: 더 구체적인 패턴이 먼저 와야 함.

        NOTE: \\b 대신 (?<!\\d)/(?!\\d)를 사용 — 한글 옆 숫자에서
        \\b가 동작하지 않는 문제 방지 (예: "주민번호 030215-4123456을")
        """
        # 숫자 경계: 숫자가 아닌 문자 또는 문자열 시작/끝
        NB_L = r'(?<!\d)'  # 왼쪽에 숫자 없음
        NB_R = r'(?!\d)'   # 오른쪽에 숫자 없음

        patterns = []

        # 1) 주민등록번호: 6자리-7자리 (앞 2자리만 보존, 나머지 마스킹)
        #    하이픈 포함: 900101-1234567 → 90****-*******
        #    연속형:      9001011234567  → 90***********
        if valves.MASK_RRN:
            # 하이픈 포함 형식
            patterns.append((
                "주민등록번호",
                re.compile(
                    NB_L +
                    r'(\d{2})'                                    # group1: 앞 2자리 (연도)
                    r'(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])'  # 월일 4자리
                    r'[\s]*[-–][\s]*'                               # 구분자
                    r'[1-4]\d{6}' +                                 # 뒷 7자리
                    NB_R
                ),
                lambda m: m.group(1) + '****-*******'
            ))
            # 연속형 (13자리, 하이픈 없음)
            patterns.append((
                "주민등록번호",
                re.compile(
                    NB_L +
                    r'(\d{2})'                                    # group1: 앞 2자리 (연도)
                    r'(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])'  # 월일 4자리
                    r'[1-4]\d{6}' +                                # 뒷 7자리 (하이픈 없이 바로 연결)
                    NB_R
                ),
                lambda m: m.group(1) + '***********'
            ))

        # 2) 운전면허번호: XX-XX-XXXXXX-XX (지역코드 보존)
        #    11-12-123456-78 → 11-**-******-**
        if valves.MASK_DRIVER_LICENSE:
            patterns.append((
                "운전면허번호",
                re.compile(
                    NB_L +
                    r'(\d{2})[-–](\d{2})[-–](\d{6})[-–](\d{2})' +
                    NB_R
                ),
                r'\1-**-******-**'
            ))

        # 3) 신용카드번호: 4자리-4자리-4자리-4자리 (앞 4자리 보존)
        #    1234-5678-9012-3456 → 1234-****-****-****
        if valves.MASK_CARD:
            patterns.append((
                "신용카드번호",
                re.compile(
                    NB_L +
                    r'(\d{4})[\s]*[-–\s][\s]*(\d{4})'
                    r'[\s]*[-–\s][\s]*(\d{4})'
                    r'[\s]*[-–\s][\s]*(\d{4})' +
                    NB_R
                ),
                r'\1-****-****-****'
            ))

        # 4) 사업자등록번호: XXX-XX-XXXXX
        #    123-45-67890 → ***-**-*****
        if valves.MASK_BRN:
            patterns.append((
                "사업자등록번호",
                re.compile(
                    NB_L + r'(\d{3})[-–](\d{2})[-–](\d{5})' + NB_R
                ),
                r'***-**-*****'
            ))

        # 5) 휴대폰/전화번호 (계좌번호보다 먼저 매칭해야 함)
        #    010-1234-5678 → 010-****-****
        #    01012345678   → 010****5678
        #    02-1234-5678  → 02-****-****
        if valves.MASK_PHONE:
            # 하이픈 포함 형식
            patterns.append((
                "전화번호",
                re.compile(
                    NB_L + r'(0\d{1,2})[-–](\d{3,4})[-–](\d{4})' + NB_R
                ),
                lambda m: f"{m.group(1)}-{'*' * len(m.group(2))}-****"
            ))
            # 하이픈 없는 형식 (010으로 시작하는 11자리)
            patterns.append((
                "전화번호",
                re.compile(
                    NB_L + r'(01[016789])(\d{3,4})(\d{4})' + NB_R
                ),
                lambda m: f"{m.group(1)}{'*' * len(m.group(2))}****"
            ))

        # 6) 계좌번호: 다양한 은행 형식 (하이픈 포함, 10~14자리)
        #    110-123-456789 → ***-***-******
        #    전화번호 패턴 이후에 배치하여 전화번호가 먼저 매칭되도록 함
        if valves.MASK_ACCOUNT:
            patterns.append((
                "계좌번호",
                re.compile(
                    NB_L + r'(\d{2,6})[-–](\d{2,6})[-–](\d{2,8})' + NB_R
                ),
                lambda m: '-'.join('*' * len(g) for g in m.groups())
            ))

        # 7) 여권번호: 알파벳 1-2자 + 7-8자리
        #    M12345678 → M********
        if valves.MASK_PASSPORT:
            patterns.append((
                "여권번호",
                re.compile(
                    r'\b([A-Z]{1,2})(\d{7,8})\b'
                ),
                lambda m: m.group(1)[0] + '*' * (len(m.group(0)) - 1)
            ))

        # 8) 이메일 주소 (로컬 앞2자리 + 도메인 앞2자리만 보존)
        #    hong.gildong@company.co.kr → ho**********@co************
        if valves.MASK_EMAIL:
            patterns.append((
                "이메일",
                re.compile(
                    r'\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
                ),
                lambda m: (
                    m.group(1)[:2] + '*' * max(len(m.group(1)) - 2, 0)
                    + '@'
                    + m.group(2)[:2] + '*' * max(len(m.group(2)) - 2, 0)
                ) if len(m.group(1)) > 2 else (
                    m.group(1)[:1] + '*' * max(len(m.group(1)) - 1, 0)
                    + '@'
                    + m.group(2)[:2] + '*' * max(len(m.group(2)) - 2, 0)
                )
            ))

        # 9) IP 주소 (기본 OFF - 코드/설정에서 자주 쓰이므로)
        #    192.168.1.100 → 192.168.***.***
        if valves.MASK_IP:
            patterns.append((
                "IP주소",
                re.compile(
                    r'\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b'
                ),
                r'\1.\2.***.***'
            ))

        return patterns

    # ──────────────────────────────────────────
    # 마스킹 엔진
    # ──────────────────────────────────────────

    def _mask_text(self, text: str) -> tuple[str, list[str]]:
        """
        텍스트에서 PII를 마스킹하고, 감지된 PII 유형 목록을 반환.
        Returns: (마스킹된 텍스트, [감지된 PII 유형 이름들])
        """
        if not text or not isinstance(text, str):
            return text, []

        detected = []
        masked = text

        for name, pattern, replacement in self._get_patterns(self.valves):
            if callable(replacement):
                new_text = pattern.sub(replacement, masked)
            else:
                new_text = pattern.sub(replacement, masked)

            if new_text != masked:
                detected.append(name)
                masked = new_text

        return masked, detected

    def _mask_messages(self, messages: list[dict]) -> list[str]:
        """
        메시지 리스트의 content를 마스킹.
        Returns: 감지된 PII 유형 목록
        """
        all_detected = []

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                masked, detected = self._mask_text(content)
                if detected:
                    msg["content"] = masked
                    all_detected.extend(detected)
            elif isinstance(content, list):
                # 멀티모달 메시지 (text + image 등)
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        masked, detected = self._mask_text(part.get("text", ""))
                        if detected:
                            part["text"] = masked
                            all_detected.extend(detected)

        return list(set(all_detected))

    # ──────────────────────────────────────────
    # Filter 인터페이스 (inlet / outlet)
    # ──────────────────────────────────────────

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """
        사용자 입력 마스킹 (LLM 전달 전)
        """
        if not self.valves.ENABLE_INLET:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        detected = self._mask_messages(messages)

        if detected and self.valves.LOG_MASKED:
            user_id = __user__.get("id", "unknown") if __user__ else "unknown"
            log.info(
                f"[PII Filter] inlet 마스킹 - user: {user_id}, "
                f"감지 유형: {', '.join(detected)}"
            )

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """
        LLM 응답 마스킹 (사용자 전달 전)
        """
        if not self.valves.ENABLE_OUTLET:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        detected = self._mask_messages(messages)

        if detected and self.valves.LOG_MASKED:
            user_id = __user__.get("id", "unknown") if __user__ else "unknown"
            log.info(
                f"[PII Filter] outlet 마스킹 - user: {user_id}, "
                f"감지 유형: {', '.join(detected)}"
            )

        return body
