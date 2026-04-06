# HRIS AI Analyzer API

AI-powered HRIS REST API untuk menganalisa interview, screening CV, dan rekomendasi lowongan kerja menggunakan Groq AI (Llama 3.3-70B).

## Fitur

- 3 endpoint AI: Interview Analyzer, CV Screening (ATS), Job Recommender
- Fire & forget pattern — response langsung, AI proses di background
- Hasil disimpan otomatis ke PostgreSQL (Railway)
- GET endpoint untuk ambil hasil berdasarkan `applicationId`
- Groq API key fallback — otomatis switch ke key cadangan jika rate limit
- CORS enabled untuk semua origin
- Deployed di Railway (auto-deploy dari GitHub)

## Arsitektur & Flow

```
Client hit POST endpoint → API return success langsung (tidak menunggu AI selesai)
                              ↓ (background task)
                 DELETE record lama (per applicationId) + INSERT status = PROCESSING
                              ↓
                 AI proses via Groq API (Llama 3.3-70B)
                              ↓
                 UPDATE DB → status = COMPLETED + result (JSON)
                          → status = FAILED + error message (jika gagal)

Client hit GET endpoint → ambil hasil berdasarkan applicationId
```

## Tech Stack

- Python 3.11 + FastAPI + Uvicorn
- Groq API (Llama 3.3-70B Versatile) dengan 3-key fallback
- PostgreSQL (Railway)
- PyMuPDF (ekstraksi teks PDF)

## Setup Lokal

1. Clone repo
```bash
git clone https://github.com/rmfirza/hris_ai_analyzer.git
cd hris_ai_analyzer
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Buat file `.env`
```env
GROQ_API_KEY_1=your_groq_api_key_1
GROQ_API_KEY_2=your_groq_api_key_2
GROQ_API_KEY_3=your_groq_api_key_3
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

## Setup Railway (Production)

1. Buka [railway.app](https://railway.app) → login
2. Buka project yang sudah ada (atau buat baru)
3. Klik **"+ New"** → **"GitHub Repo"** → pilih repo `rmfirza/hris_ai_analyzer`
4. Set branch: `interview-analyzer-agent-ai`, root directory: `/interview-analyzer-agent-ai`
5. Tambahkan **Variables** di tab Variables:
   - `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, `GROQ_API_KEY_3`
   - `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
6. Buka **Settings** → **Networking** → klik **Generate Domain**
7. Railway akan otomatis deploy dan redeploy setiap ada push ke GitHub

## Base URL (Production)

```
https://hrisaianalyzer-production.up.railway.app
```

---

## Endpoints

### `GET /`
Health check.

**Response:**
```json
{ "status": "ok, LLAMA 3.3 Engine Online!" }
```

---

### `POST /analyze-interview`
Analisa transkrip wawancara kandidat. Content-Type: `application/json`

**Request Body:**
```json
{
  "applicationId": "APP_123",
  "jobId": "JOB_456",
  "roomName": "room-abc",
  "roomSid": "sid-xyz",
  "endedAt": "2026-04-03T10:00:00.000Z",
  "transcript": [
    { "role": "assistant", "text": "Perkenalkan diri kamu", "createdAt": 1700000000 },
    { "role": "user", "text": "Nama saya Budi, saya backend engineer", "createdAt": 1700000010 }
  ]
}
```

| Field | Type | Wajib | Keterangan |
|---|---|---|---|
| applicationId | string | Ya | ID unik aplikasi |
| jobId | string | Ya | ID lowongan |
| roomName | string | Tidak | Nama room interview |
| roomSid | string | Tidak | Session ID room |
| endedAt | string (ISO 8601) | Tidak | Waktu interview selesai |
| transcript | array | Ya | Minimal 1 item, tidak boleh kosong |

**Response:**
```json
{
  "status": "success",
  "message": "Diproses Llama AI di background.",
  "applicationId": "APP_123",
  "jobId": "JOB_456"
}
```

---

### `GET /result/interview/{application_id}`
Ambil hasil analisa interview berdasarkan `applicationId`.

**Response:**
```json
{
  "application_id": "APP_123",
  "status": "COMPLETED",
  "data": {
    "score_breakdown": {
      "communication": { "question_asked": "...", "candidate_answer": "...", "score": 7, "strong_signal": "...", "red_flag": "..." },
      "technical": { "question_asked": "...", "candidate_answer": "...", "score": 8, "strong_signal": "...", "red_flag": "..." },
      "problem_solving": { "question_asked": "...", "candidate_answer": "...", "score": 6, "strong_signal": "...", "red_flag": "..." },
      "culture_fit": { "question_asked": "...", "candidate_answer": "...", "score": 7, "strong_signal": "...", "red_flag": "..." }
    },
    "psychological_profile": {
      "personality_summary": "...",
      "estimated_mbti": "INTJ",
      "mbti_reasoning": "...",
      "estimated_disc": "High C, Low I",
      "disc_reasoning": "..."
    },
    "ai_evaluation_insight": {
      "key_strengths": ["...", "..."],
      "growth_areas": ["...", "..."]
    },
    "machine_recommendation": "RECOMMENDED FOR HIRE | CONSIDER | REJECT",
    "executive_summary": "...",
    "ai_processing_time_seconds": 4.2
  },
  "transcript": [
    { "role": "assistant", "text": "...", "createdAt": 1700000000 },
    { "role": "user", "text": "...", "createdAt": 1700000010 }
  ]
}
```

---

### `POST /analyze-cv-employer`
Screening CV kandidat terhadap job description (ATS). Content-Type: `multipart/form-data`

| Field | Type | Wajib | Keterangan |
|---|---|---|---|
| applicationId | string | Ya | ID unik aplikasi |
| jobId | string | Ya | ID lowongan |
| jobTitle | string | Ya | Nama posisi |
| jobRequirements | string | Ya | Deskripsi requirement |
| jobIndustry | string | Tidak | Industri perusahaan |
| cvFile | file | Ya | Harus format PDF |

**Response:**
```json
{
  "status": "success",
  "message": "Diproses Llama AI di background.",
  "applicationId": "APP_123",
  "jobId": "JOB_456"
}
```

---

### `GET /result/cv-employer/{application_id}`
Ambil hasil screening CV berdasarkan `applicationId`.

**Response:**
```json
{
  "application_id": "APP_123",
  "status": "COMPLETED",
  "data": {
    "scoring_analysis": "Core tools dibutuhkan: [X, Y, Z]. Ada di CV: [X]. Tidak ada: [Y, Z]. Proporsi: 1/3 = 33%. Kesimpulan skor: 25.",
    "match_score": 85,
    "recommendation": "Lanjut | Tidak Lanjut | Pertimbangkan",
    "ai_reason": "...",
    "matching_skills": ["Python", "FastAPI"],
    "missing_skills": ["Docker", "Kubernetes"],
    "hr_consideration": "...",
    "ai_processing_time_seconds": 1.2
  }
}
```

---

### `POST /recommend-jobs`
Rekomendasi lowongan terbaik berdasarkan CV. Content-Type: `multipart/form-data`

Hasil: **top 10 rekomendasi**, hanya yang `match_score >= 65`, diurutkan dari skor tertinggi.

| Field | Type | Wajib | Keterangan |
|---|---|---|---|
| applicationId | string | Ya | ID unik aplikasi |
| seekerName | string | Ya | Nama pencari kerja |
| jobs | string (JSON array) | Ya | Daftar lowongan dari sistem, harus valid JSON |
| cvFile | file | Ya | Harus format PDF |

Format field `jobs`:
```json
[
  { "title": "Backend Engineer", "company": "PT ABC", "industry": "Tech", "requirements": "Go, PostgreSQL, Docker" },
  { "title": "Data Engineer", "company": "PT XYZ", "industry": "Finance", "requirements": "Airflow, Spark, Python, BigQuery" }
]
```

**Response:**
```json
{
  "status": "success",
  "message": "Diproses Llama AI di background.",
  "applicationId": "APP_123"
}
```

---

### `GET /result/recommend-jobs/{application_id}`
Ambil hasil rekomendasi lowongan berdasarkan `applicationId`.

**Response:**
```json
{
  "application_id": "APP_123",
  "status": "COMPLETED",
  "data": {
    "ai_processing_time_seconds": 5.1,
    "total_recommended": 3,
    "recommendations": [
      {
        "job_title": "Backend Engineer",
        "company": "PT ABC",
        "industry": "Tech",
        "match_score": 85,
        "scoring_analysis": "Core tools dibutuhkan: [Go, PostgreSQL, Docker]. Ada di CV: [Go, PostgreSQL, Docker]. Proporsi: 3/3 = 100%.",
        "why_it_fits": "...",
        "what_to_improve": "..."
      }
    ]
  }
}
```

---

## Status Response

| Status | Keterangan |
|---|---|
| `PROCESSING` | AI masih berjalan di background |
| `COMPLETED` | Hasil tersedia di field `data` |
| `FAILED` | Terjadi error, cek field `data.error` |

## Database Tables

| Table | Key Columns |
|---|---|
| `ai_interview_analyzer_result` | id, application_id, job_id, room_id, room_name, interviewed_at, analyzed_at, status, result, transcript |
| `ai_cv_analysis_result` | id, application_id, job_id, analyzed_at, status, result |
| `ai_recommended_job_result` | id, application_id, analyzed_at, status, result |

> Setiap `applicationId` hanya menyimpan 1 record (record lama otomatis dihapus saat request baru masuk).

## Validasi

| Rule | Detail |
|---|---|
| File CV | Harus format `.pdf`, teks harus bisa diekstrak (bukan scan/gambar kosong) |
| Transcript | Tidak boleh kosong (minimal 1 item) |
| Jobs | Harus valid JSON array |
| Match score threshold | Hanya lowongan dengan `match_score >= 65` yang direkomendasikan |
