#!/bin/bash
# run_api.sh: OpenAI Key 설정 + FastAPI 서버 실행 (포트 8000)

# .env에서 불러오기
if [ -f .env ]; then
  export $(grep OPENAI_API_KEY .env | xargs)
  echo "[ℹ️] .env에서 OPENAI_API_KEY를 불러왔습니다"
else
  echo "❌ .env 파일이 없습니다. 먼저 setup_env.sh를 실행해주세요."
  exit 1
fi

# FastAPI 서버 실행 (백그라운드 + 로그 파일로 리디렉션)
echo "[🚀] FastAPI 서버 실행 중... (포트 8000)"
nohup uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > fastapi.log 2>&1 &
#uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

sleep 2

if lsof -i :8000 | grep LISTEN; then
  echo "✅ FastAPI 서버가 정상적으로 포트 8000에서 기동되었습니다."
else
  echo "❌ FastAPI 서버가 포트 8000에서 실행되지 않았습니다."
  exit 1
fi
