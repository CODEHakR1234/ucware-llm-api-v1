### 실행 방법 (로컬)

```bash
export OPENAI_API_KEY="sk-........................"
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
