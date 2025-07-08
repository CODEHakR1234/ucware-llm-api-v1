# 🧠 ucware-llm-api-v1

`ucware-llm-api-v1`는 PDF 문서와 채팅 대화의 요약 기능을 제공하는 FastAPI 기반 멀티 요약 API입니다.
LangChain 0.2+ 구조를 따르며, OpenAI GPT, Chroma 벡터 DB, Redis 캐시를 통합해 대규모 문서도 빠르고 정확하게 요약할 수 있습니다.

---

## 🌟 주요 특징

* 📄 **PDF 요약**: 웹 링크 기반 PDF 다운로드 및 요약 처리
* 💬 **채팅 대화 요약**: 시간 순 정렬된 대화 데이터 요약
* ⚙️ **최신 LangChain 구조**: `langchain-openai`, `langchain-chroma` 사용
* 🧠 **LLM 기반 요약**: OpenAI GPT-3.5 또는 GPT-4 지원
* 💾 **Chroma Vector DB**: 벡터 임베딩을 저장 및 유사도 검색
* ⚡ **Redis 캐시**: 중복 요약 요청 방지 및 응답 속도 향상

---

## 🚀 빠른 시작 (KT Cloud / 리눅스 서버 환경 기준)

### ✅ 0. 프로젝트 클론 및 실행 권한 부여

```bash
git clone https://github.com/CODEHakR1234/ucware-llm-api-v1.git
cd ucware-llm-api-v1
chmod +x scripts/*.sh
chmod +x run_all.sh
chmod +x setup_env.sh
chmod +x run_vllm.sh
```

>(vllm 모델 서버 작동) 
```bash
./run_vllm.sh
```
>실행 환경 설정
```bash
./setup_env.sh
```

> 빠른 시작
```bash
./run_all.sh
```
## ✅ 1. KT Cloud vLLM 서버 구동

KT Cloud 인스턴스에서 Hugging Face 기반 LLM을 `vLLM` 서버로 구동하여, OpenAI API 호환 인터페이스로 외부 서비스와 연결할 수 있습니다.  
이 서버는 사용자가 입력한 모델을 자동으로 로딩하며, FastAPI와 같은 애플리케이션은 이를 `http://localhost:12000/v1`으로 호출하여 사용할 수 있습니다.

### 📌 주요 특징

- **vLLM 기반 OpenAI 호환 API 서버** 백그라운드 실행
- **모델 이름 직접 입력 가능**, 기본값은 `google/gemma-7b-it`
- **Hugging Face 토큰**은 실행 시 안전하게 입력받음 (입력값 숨김 처리)
- 실행된 서버는 **포트 12000**에서 대기하며, 로그는 `vllm.log`에 저장됨
- **모델 로딩 완료까지 자동 대기** (최대 180초까지 포트 오픈 감지)

---

### ▶️ 실행 방법

```bash
./run_vllm.sh
```

### ✅ 2. 환경 설정 및 의존성 설치

프로젝트 실행을 위한 필수 패키지와 Python 가상환경을 자동으로 구성하는 단계입니다.
이 과정을 통해 Redis, OCR 도구, PDF 처리 도구, Python 의존성 등이 설치되며, `.env` 파일도 자동 생성됩니다.

#### 📌 실행 명령어

```bash
./setup_env.sh
```

#### 🛠️ 수행 작업

1. **시스템 패키지 설치**

   * `redis-server`: 캐시 기능을 제공하는 인메모리 저장소

2. **Python 가상환경 생성**

   * Python 3.10+ 기반 `.venv` 가상환경을 생성하고 활성화합니다.

3. **Python 패키지 설치**

   * `pip install -r requirements.txt`를 통해 프로젝트에 필요한 모든 Python 라이브러리를 설치합니다.

4. **환경변수 파일(.env) 생성**

   * `.env` 파일이 자동 생성되며, 관리자가 선택 가능한 llm 와 embedding 모델
/OPENAI API 키/Chroma/Redis 관련 설정을 포함합니다.

---

### ✅ 3. Redis 및 Chroma 서버 실행

백엔드에서 동작하는 Redis 캐시 서버와 Chroma 벡터 데이터베이스 서버를 실행합니다.
이 단계는 요약 요청 시 내부적으로 필요한 **벡터 검색과 중복 요청 캐싱** 기능을 활성화합니다.

#### 📌 실행 명령어

```bash
./start_services.sh
```

#### 🛠️ 수행 작업

1. **Redis 서버 실행**

   * 백그라운드에서 Redis 서버를 실행합니다.
   * 실행 확인: `redis-cli ping` → `PONG`이면 정상 작동

2. **Chroma DB 실행**

   * 벡터 저장소 역할을 하는 Chroma 서버를 포트 `9000`에서 실행합니다.
   * 실행 확인: `curl http://localhost:9000/api/v2/heartbeat` → `{"nanosecond heartbeat": ...}`

> Redis는 `127.0.0.1:6379`, Chroma는 `0.0.0.0:9000`에서 서비스됩니다.

---

### ✅ 4. OpenAI API 키 등록 및 FastAPI 실행

OpenAI의 GPT 모델을 호출하기 위해 API 키가 필요합니다.
이 키를 .env파일로 부터 등록한 뒤 FastAPI 서버를 실행하면 
`/summary` 및 `/ciat-summary` 등의 API가 활성화됩니다.

#### 📌 실행 명령어

```bash
./run_api.sh
```

#### 🛠️ 수행 작업

1. **OpenAI API 키 입력**

   * 사용자에게 키를 입력받아 `OPENAI_API_KEY` 환경 변수로 등록합니다.

2. **FastAPI 서버 실행**

   * 포트 `8000`에서 FastAPI 서버를 `--reload` 모드로 실행합니다.
   * 자동 문서화: [http://localhost:8000/docs](http://localhost:8000/docs) 접속 가능

#### 💡 결과

* 로컬에서 `POST /summary`, `POST /chat-summary` 등 API를 사용할 수 있게 됩니다.
* `.venv` 가상환경이 자동 활성화된 상태에서 서버가 실행됩니다.

---

### ✅ 5. 서버 종료

Redis, Chroma, FastAPI 서버가 백그라운드 혹은 터미널에서 실행되고 있다면, 이 명령어로 한 번에 모두 종료할 수 있습니다.

#### 📌 실행 명령어

```bash
./stop_services.sh
```

#### 🛠️ 수행 작업

1. **현재 열린 포트 확인**

   * `lsof -i :6379`, `:9000`, `:8000` 포트를 통해 Redis, Chroma, API 서버 PID 확인

2. **프로세스 종료**

   * 각 포트를 사용하는 PID를 `kill` 명령으로 안전하게 종료합니다.

#### ✅ 효과

* 모든 서버 리소스가 해제되어 재시작 시 충돌 없음
* Chroma가 다시 실행될 때 동일한 `/chroma_db` 경로로 복원 가능
* Redis는 다시 시작 시 TTL 설정에 따라 캐시 재사용 가능

---
### ✅ 6. 전체 통합 실행 스크립트

한 번의 명령어로 전체 실행 과정을 자동화하고 싶다면 다음 스크립트를 사용할 수 있습니다.

#### 📌 실행 명령어

```bash
./run_all.sh

