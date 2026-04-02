# How to Start — HRIS AI Analyzer API

## Prerequisites
- Python 3.9+
- Ngrok account (static domain)
- Groq API Key
- Railway PostgreSQL (sudah setup)

---

## 1. Clone Repo
```bash
git clone https://github.com/rmfirza/hris_ai_analyzer.git
cd hris_ai_analyzer
git checkout interview-analyzer-agent-ai
```

---

## 2. Install Dependencies
```bash
pip install fastapi uvicorn groq python-dotenv psycopg2-binary pymupdf
```

---

## 3. Buat File `.env`
Buat file `.env` di folder yang sama dengan `api_analyzer.py`:
```env
GROQ_API_KEY=your_groq_api_key

DB_HOST=interchange.proxy.rlwy.net
DB_PORT=11544
DB_NAME=railway
DB_USER=postgres
DB_PASSWORD=your_db_password
```

---

## 4. Jalankan Server
```bash
uvicorn api_analyzer:app --reload
```

Kalau berhasil terminal bakal nunjukin:
```
[DB] Semua table siap.
INFO: Uvicorn running on http://127.0.0.1:8000
```

---

## 5. Jalankan Ngrok
Buka terminal baru, jalankan:
```bash
ngrok http --url=unmobilized-unopted-kenneth.ngrok-free.dev 8000
```

---

## API Siap Di-hit

| Endpoint | Method | Content-Type |
|---|---|---|
| `/analyze-interview` | POST | application/json |
| `/analyze-cv-employer` | POST | multipart/form-data |
| `/recommend-jobs` | POST | multipart/form-data |

Base URL: `https://unmobilized-unopted-kenneth.ngrok-free.dev`

---

## Cek Dokumentasi Interaktif
Buka di browser:
```
http://localhost:8000/docs
```

---

## Database
Hasil analisa tersimpan otomatis di Railway PostgreSQL:

| Endpoint | Table |
|---|---|
| `/analyze-interview` | `ai_interview_analyzer_result` |
| `/analyze-cv-employer` | `ai_cv_analysis_result` |
| `/recommend-jobs` | `ai_recommended_job_result` |
