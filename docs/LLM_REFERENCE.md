# LLM Reference - Open WebUI 로컬 개발 환경

이 문서는 LLM(AI 어시스턴트)이 이 프로젝트 작업 시 참고할 수 있도록 작성되었다.

## 프로젝트 개요

- **프로젝트**: Open WebUI v0.8.3
- **설치 방식**: 소스코드 기반 로컬 설치 (Docker 미사용)
- **설치 일자**: 2026-02-19
- **소스 원본**: https://github.com/open-webui/open-webui (main 브랜치)

## 기술 스택

| 구분 | 기술 | 버전 |
|------|------|------|
| 프론트엔드 | SvelteKit + Vite | package.json 참조 |
| 백엔드 | Python + FastAPI | Python 3.11.7 |
| DB | SQLite | 기본 내장 |
| 패키지 매니저 (FE) | npm | 11.7.0 |
| 패키지 매니저 (BE) | pip | venv 내장 |
| Node.js | v22.22.0 | winget portable 설치 |

## 디렉토리 구조 (주요)

```
d:\myproject\my_openwebui\
├── src/                    # SvelteKit 프론트엔드 소스
│   ├── lib/                # 컴포넌트, 유틸, 스토어
│   └── routes/             # 페이지 라우팅
├── backend/                # FastAPI 백엔드
│   ├── open_webui/         # 메인 애플리케이션 코드
│   │   ├── main.py         # FastAPI 앱 진입점
│   │   ├── routers/        # API 라우터
│   │   ├── models/         # DB 모델
│   │   └── utils/          # 유틸리티
│   ├── venv/               # Python 가상환경 (gitignore됨)
│   └── requirements.txt    # Python 의존성
├── build/                  # 프론트엔드 빌드 결과물 (gitignore됨)
├── node_modules/           # npm 패키지 (gitignore됨)
├── static/                 # 정적 파일 (pyodide 등)
├── .env                    # 환경 설정 (.env.example에서 복사)
└── docs/                   # 문서
```

## 서버 실행 방법

### 프로덕션 모드 (빌드된 프론트엔드 + 백엔드 통합)

```bash
cd d:\myproject\my_openwebui\backend
set PYTHONIOENCODING=utf-8
venv\Scripts\activate
uvicorn open_webui.main:app --host 0.0.0.0 --port 8080
```

- 접속: http://localhost:8080
- 프론트엔드는 `build/` 디렉토리의 정적 파일을 백엔드가 서빙

### 개발 모드 (프론트엔드 핫 리로드)

터미널 1 - 백엔드:
```bash
cd d:\myproject\my_openwebui\backend
set PYTHONIOENCODING=utf-8
venv\Scripts\activate
uvicorn open_webui.main:app --reload --host 0.0.0.0 --port 8080
```

터미널 2 - 프론트엔드:
```bash
cd d:\myproject\my_openwebui
npm run dev
```

- 프론트엔드 접속: http://localhost:5173 (Vite 개발 서버, API는 8080으로 프록시)
- 백엔드 접속: http://localhost:8080

## 빌드 방법

### 프론트엔드 재빌드
```bash
cd d:\myproject\my_openwebui
npm run build
```
- `--legacy-peer-deps` 옵션이 필요할 수 있음 (`npm install` 시)
- @tiptap/core v2와 v3 간의 peer dependency 충돌 존재

### 백엔드 의존성 재설치
```bash
cd d:\myproject\my_openwebui\backend
venv\Scripts\activate
pip install -r requirements.txt -U
```

## Windows 환경 주의사항

### 1. cp949 인코딩 크래시 (필수)
`backend/open_webui/main.py:579`에 ASCII 아트 배너가 유니코드 블록 문자(`█`, `╗` 등)를 사용한다.
한국어 Windows의 기본 인코딩(cp949)으로는 이 문자를 출력할 수 없어 서버가 크래시한다.

**해결**: 서버 실행 전 반드시 `PYTHONIOENCODING=utf-8` 환경변수를 설정해야 한다.
```bash
# CMD
set PYTHONIOENCODING=utf-8

# PowerShell
$env:PYTHONIOENCODING = "utf-8"

# Git Bash
export PYTHONIOENCODING=utf-8
```

### 2. npm peer dependency 충돌
`npm install` 시 @tiptap 패키지 간 버전 충돌이 발생한다.
```bash
npm install --legacy-peer-deps
```

### 3. Node.js 버전 요구사항
일부 의존성(yargs@18)이 Node.js `>=22.12.0`을 요구한다.
현재 Node.js 22.22.0이 winget portable로 설치되어 있다.

경로: `C:\Users\user\AppData\Local\Microsoft\WinGet\Packages\OpenJS.NodeJS.22_Microsoft.Winget.Source_8wekyb3d8bbwe\node-v22.22.0-win-x64\`

기존 `C:\Program Files\nodejs\` 에도 v22.11.0이 남아 있으므로, PATH 순서에 주의해야 한다.

### 4. Huggingface symlink 경고
모델 다운로드 시 `WinError 1314` (권한 부족) 경고가 발생할 수 있다.
Windows 개발자 모드를 켜거나 관리자 권한이 필요하지만, 기능 자체는 정상 동작한다.

## 데이터 저장 위치

| 데이터 | 경로 |
|--------|------|
| SQLite DB | `~/.open-webui/webui.db` |
| 업로드 파일 | `~/.open-webui/uploads/` |
| 캐시 | `~/.open-webui/cache/` |
| HuggingFace 모델 | `~/.cache/huggingface/` |

## 환경 설정 (.env)

`.env.example`에서 복사하여 생성했으며, 주요 설정:
- `OLLAMA_BASE_URL`: Ollama 서버 주소 (기본: http://localhost:11434)
- `OPENAI_API_BASE_URL` / `OPENAI_API_KEY`: OpenAI 호환 API 설정
- `CORS_ALLOW_ORIGIN`: CORS 허용 (개발 시 `*`)

전체 환경변수 목록은 공식 문서 참조: https://docs.openwebui.com/

## 플랫폼 서비스 (Docker Compose)

Open WebUI 외 서비스들은 `d:\myproject\platform\`에서 Docker Compose로 관리한다.

### 디렉토리 구조

```
d:\myproject\platform\
├── docker-compose.yaml    # 전체 서비스 오케스트레이션
├── .env                   # 공용 환경변수 (DB, Redis 등)
├── init-db.sql            # 서비스별 DB 자동 생성 스크립트
├── outline/
│   └── .env               # Outline Wiki 전용 설정
└── data/
    ├── postgres/           # PostgreSQL 데이터 영속화
    ├── redis/              # Redis 데이터
    └── minio/              # MinIO 파일 스토리지
```

### 서비스 목록

| 서비스 | 접속 주소 | 실행 방식 | 비고 |
|--------|-----------|-----------|------|
| Open WebUI | `localhost:8080` | 소스 직접 실행 | `backend/venv` 사용 |
| Outline Wiki | `localhost:3000` | Docker | OIDC로 Google 로그인 |
| PostgreSQL 16 | `localhost:5432` | Docker (공용) | 서비스별 DB 분리 |
| Redis 7 | `localhost:6379` | Docker (공용) | |
| MinIO | `localhost:9000` (API) / `localhost:9001` (콘솔) | Docker | Outline 파일 스토리지 |

### 실행/중지

```bash
cd d:\myproject\platform

# 전체 기동
docker compose up -d

# 전체 중지
docker compose down

# 특정 서비스만 기동
docker compose up -d outline

# 로그 확인
docker compose logs -f outline
```

### Outline Wiki 인증

- Outline은 기본 Google OAuth에서 **개인 Gmail을 차단**한다 (하드코딩)
- 해결: Google OAuth 대신 **OIDC로 Google 연동** (`outline/.env` 참조)
- Google Cloud Console에서 리디렉션 URI에 `http://localhost:3000/auth/oidc.callback` 등록 필요

### 서비스 추가 방법

새 서비스(Dify, n8n 등) 추가 시:
1. `init-db.sql`에 `CREATE DATABASE <서비스명>;` 추가
2. `docker-compose.yaml`에 서비스 블록 추가 (네트워크: `platform-net`)
3. `platform/<서비스명>/.env` 생성
4. 같은 Docker 네트워크 내이므로 컨테이너 이름으로 통신 가능 (예: `http://postgres:5432`)

### 주의사항

- `docker compose restart`는 환경변수를 다시 읽지 않음. `.env` 변경 시 `docker compose up -d --force-recreate <서비스>` 사용
- PostgreSQL 데이터는 `platform/data/postgres/`에 영속화됨. `docker compose down`으로 중지해도 데이터 유지
- `docker compose down -v` 사용 시 볼륨까지 삭제되므로 주의

## 코드 수정 시 참고

- 프론트엔드 수정 후 `npm run build` 필요 (프로덕션 모드 시)
- 개발 모드(`npm run dev`)에서는 핫 리로드 지원
- 백엔드 `--reload` 옵션 사용 시 코드 변경 자동 반영
- DB 스키마 변경은 Alembic 마이그레이션 사용 (`backend/open_webui/migrations/`)
