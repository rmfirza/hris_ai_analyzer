# HRIS AI Analyzer API

AI-powered HRIS REST API untuk menganalisa interview, screening CV, dan rekomendasi lowongan kerja menggunakan Groq AI (Llama 3.3-70B).

## Fitur

- 3 endpoint AI: Interview Analyzer, CV Screening (ATS), Job Recommender
- Fire & forget pattern — response langsung, AI proses di background
- Hasil disimpan otomatis ke PostgreSQL (Railway)
- Deployed di Railway (auto-deploy dari GitHub)

## Arsitektur & Flow

```
Client hit endpoint → API return success langsung (tidak menunggu AI selesai)
                         ↓ (background task)
              INSERT ke DB dengan status = PROCESSING
                         ↓
              AI proses via Groq API (Llama 3.3-70B)
                         ↓
              UPDATE DB → status = COMPLETED + result (JSON)
                       → status = FAILED + error message (jika gagal)
```

**Cara ambil hasil:** Client query langsung ke database berdasarkan `application_id`.

## Tech Stack

- Python 3.11 + FastAPI + Uvicorn
- Groq API (Llama 3.3-70B Versatile)
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

## Setup Railway (Production)

1. Buka [railway.app](https://railway.app) → login
2. Buka project yang sudah ada (atau buat baru)
3. Klik **"+ New"** → **"GitHub Repo"** → pilih repo `rmfirza/hris_ai_analyzer`
4. Set branch: `interview-analyzer-agent-ai`, root directory: `/interview-analyzer-agent-ai`
5. Tambahkan **Variables** di tab Variables:
   - `GROQ_API_KEY`
   - `DB_HOST`
   - `DB_PORT`
   - `DB_NAME`
   - `DB_USER`
   - `DB_PASSWORD`
6. Buka **Settings** → **Networking** → klik **Generate Domain**
7. Railway akan otomatis deploy dan redeploy setiap ada push ke GitHub

## Base URL (Production)

```
https://hrisaianalyzer-production.up.railway.app
```

## Endpoints

### `GET /`
Health check.

**Response:**
```json
{ "status": "ok" }
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
  "message": "Data berhasil diterima. AI sedang menganalisa di background.",
  "jobId": "JOB_456"
}
```

**AI Output (disimpan ke DB):**

Scoring per kategori: **1-10**

```json
{
  "score_breakdown": {
    "communication": { "score": 7, "evidence": "...", "strong_signal": "...", "red_flag": "None" },
    "technical": { "score": 8, "evidence": "...", "strong_signal": "...", "red_flag": "None" },
    "problem_solving": { "score": 6, "evidence": "...", "strong_signal": "...", "red_flag": "..." },
    "culture_fit": { "score": 7, "evidence": "...", "strong_signal": "...", "red_flag": "None" }
  },
  "ai_evaluation_insight": {
    "key_strengths": ["...", "..."],
    "growth_areas": ["...", "..."]
  },
  "machine_recommendation": "RECOMMENDED_FOR_HIRE | CONSIDER | REJECT",
  "executive_summary": "..."
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
  "message": "Data berhasil diterima. AI sedang menganalisa di background.",
  "applicationId": "APP_123",
  "jobId": "JOB_456"
}
```

**AI Output (disimpan ke DB):**

`match_score`: **0-100**

```json
{
  "match_score": 85,
  "recommendation": "Lanjut | Tidak Lanjut | Pertimbangkan",
  "ai_reason": "...",
  "matching_skills": ["Python", "FastAPI"],
  "missing_skills": ["Docker"],
  "hr_consideration": "..."
}
```

---

### `POST /recommend-jobs`
Rekomendasi lowongan terbaik berdasarkan CV. Content-Type: `multipart/form-data`

Hasil: **top 10 rekomendasi**, hanya yang `match_score >= 50`.

| Field | Type | Wajib | Keterangan |
|---|---|---|---|
| applicationId | string | Ya | ID unik aplikasi |
| seekerName | string | Ya | Nama pencari kerja |
| jobs | string (JSON array) | Ya | Daftar lowongan, harus valid JSON |
| cvFile | file | Ya | Harus format PDF |

Format `jobs`:
```json
[
  { "title": "Backend Engineer", "company": "PT ABC", "industry": "Tech", "requirements": "Go, PostgreSQL" },
  { "title": "Data Analyst", "company": "PT XYZ", "industry": "Finance", "requirements": "SQL, Python" }
]
```

**Response:**
```json
{
  "status": "success",
  "message": "Data berhasil diterima. AI sedang mencari rekomendasi lowongan di background.",
  "applicationId": "APP_123"
}
```

**AI Output (disimpan ke DB):**

`match_score`: **0-100**, difilter `>= 50`, diurutkan tertinggi, maks 10.

```json
[
  {
    "job_title": "Backend Engineer",
    "company": "PT ABC",
    "industry": "Tech",
    "match_score": 85,
    "why_it_fits": "...",
    "what_to_improve": "..."
  }
]
```

## Database Tables

| Table | Key Columns |
|---|---|
| `ai_interview_analyzer_result` | id, application_id, job_id, room_id, room_name, interviewed_at, analyzed_at, status, result |
| `ai_cv_analysis_result` | id, application_id, job_id, analyzed_at, status, result |
| `ai_recommended_job_result` | id, application_id, analyzed_at, status, result |

**Status flow:** `PROCESSING` → `COMPLETED` (result berisi JSON output AI) / `FAILED` (result berisi error message)

## Validasi

| Rule | Detail |
|---|---|
| File CV | Harus format `.pdf`, teks harus bisa diekstrak (bukan scan/gambar kosong) |
| Transcript | Tidak boleh kosong (minimal 1 item) |
| Jobs | Harus valid JSON array |
