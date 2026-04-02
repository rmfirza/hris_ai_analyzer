# Interview Analyzer Agent AI

AI-powered REST API untuk menganalisa transkrip wawancara kandidat secara otomatis menggunakan Groq API.

## Fitur

- Terima transkrip wawancara via POST request
- Response cepat (fire & forget) — tidak perlu menunggu proses AI selesai
- Analisa berjalan di background, hasil disimpan ke PostgreSQL
- Scoring 4 kategori: Communication, Technical, Problem Solving, Culture Fit

## Tech Stack

- Python
- FastAPI + Uvicorn
- Groq API (Llama 3.3-70B)
- PostgreSQL (Railway)

## Setup

1. Clone repo & masuk ke folder
```bash
git clone https://github.com/rmfirza/hris_ai_analyzer.git
cd hris_ai_analyzer
```

2. Install dependencies
```bash
pip install fastapi uvicorn groq python-dotenv psycopg2-binary
```

3. Buat file `.env`
```env
GROQ_API_KEY=your_groq_api_key

DB_HOST=your_db_host
DB_PORT=your_db_port
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
```

4. Jalankan server
```bash
uvicorn api_analyzer:app --reload
```

## Endpoint

### POST `/analyze-interview`

**Request Body:**
```json
{
  "ApplicationId": "app_123",
  "jobId": "job_456",
  "roomName": "voice_assistant_room_001",
  "roomSid": "RM_xxxx",
  "endedAt": "2026-04-02T02:51:43.741Z",
  "transcript": [
    { "role": "assistant", "text": "Pertanyaan interviewer", "createdAt": 1700000000 },
    { "role": "user", "text": "Jawaban kandidat", "createdAt": 1700000010 }
  ]
}
```

**Response (langsung):**
```json
{
  "status": "success",
  "message": "Transcript diterima, analisa sedang berjalan di background.",
  "roomName": "voice_assistant_room_001",
  "jobId": "job_456"
}
```

## Struktur Tabel PostgreSQL

| Kolom | Tipe |
|---|---|
| id | UUID |
| application_id | VARCHAR |
| job_id | VARCHAR |
| room_id | VARCHAR |
| room_name | VARCHAR |
| interviewed_at | TIMESTAMP |
| analyzed_at | TIMESTAMP |
| result | JSONB |
