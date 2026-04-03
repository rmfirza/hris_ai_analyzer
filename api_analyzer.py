import os
import io
import json
import uuid
import fitz
import psycopg2
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

# ==========================================
# 1. SETUP ENVIRONMENT & API CLIENT
# ==========================================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

LLAMA_MODEL = "llama-3.3-70b-versatile"

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

app = FastAPI(title="HRIS AI Analyzer API", version="3.0")

# ==========================================
# 2. SETUP TABLES
# ==========================================
def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_interview_analyzer_result (
            id              UUID PRIMARY KEY,
            application_id  VARCHAR(255),
            job_id          VARCHAR(255),
            room_id         VARCHAR(255),
            room_name       VARCHAR(255),
            interviewed_at  VARCHAR(255),
            analyzed_at     TIMESTAMP,
            status          VARCHAR(50),
            result          JSONB
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_cv_analysis_result (
            id              UUID PRIMARY KEY,
            application_id  VARCHAR(255),
            job_id          VARCHAR(255),
            analyzed_at     TIMESTAMP,
            status          VARCHAR(50),
            result          JSONB
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_recommended_job_result (
            id              UUID PRIMARY KEY,
            application_id  VARCHAR(255),
            analyzed_at     TIMESTAMP,
            status          VARCHAR(50),
            result          JSONB
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[DB] Semua table siap.")

init_db()

# ==========================================
# 3. HELPER: EXTRACT TEKS DARI PDF BINARY
# ==========================================
def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    stream = io.BytesIO(pdf_bytes)
    text = ""
    with fitz.open(stream=stream, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# ==========================================
# 4. SKEMA JSON (INTERVIEW)
# ==========================================
class TranscriptItem(BaseModel):
    role: str
    text: str
    createdAt: int

class InterviewPayload(BaseModel):
    applicationId: Optional[str] = None
    jobId: str
    roomName: Optional[str] = None
    roomSid: Optional[str] = None
    endedAt: Optional[str] = None
    transcript: List[TranscriptItem]

# ==========================================
# 5. BACKGROUND: ANALYZE INTERVIEW
# ==========================================
def process_interview_background(record_id: str, payload: InterviewPayload):
    print(f"\n[BACKGROUND] Mulai analisa interview Job ID: {payload.jobId}...")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ai_interview_analyzer_result
        (id, application_id, job_id, room_id, room_name, interviewed_at, status, result)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (record_id, payload.applicationId, payload.jobId, payload.roomSid, payload.roomName, payload.endedAt, "PROCESSING", None))
    conn.commit()
    cur.close()
    conn.close()
    print(f"[DB] Record {record_id} — status: PROCESSING")

    conversation_history = ""
    for chat in payload.transcript:
        role = "Interviewer" if chat.role == "assistant" else "Candidate"
        conversation_history += f"{role}: {chat.text}\n"

    prompt_analyzer = f"""
    Lu adalah AI Engineering Manager & Senior Technical Recruiter yang SANGAT TELITI dan ANTI-BIAS.
    Tugas lu menganalisa TRANSKRIP WAWANCARA KANDIDAT dan memberikan penilaian objektif beserta BUKTI (Evidence).

    TRANSKRIP WAWANCARA:
    {conversation_history}

    === RUBRIK PENILAIAN MUTLAK ===
    Berikan skor (1-10) untuk setiap kategori:
    1. COMMUNICATION: Ide terstruktur, menjawab langsung, kejelasan bahasa.
    2. TECHNICAL: Pemahaman fundamental, penjelasan keputusan teknis, best practice.
    3. PROBLEM SOLVING: Pendekatan sistematis, kemandirian debugging.
    4. CULTURE FIT: Teamwork, pragmatis, cara terima feedback.

    === ATURAN FORMAT OUTPUT ===
    Wajib sertakan: "score", "evidence" (kutipan langsung), "strong_signal", dan "red_flag".

    Return ONLY valid JSON dengan format ini:
    {{
        "score_breakdown": {{
            "communication": {{ "score": 0, "evidence": "", "strong_signal": "", "red_flag": "" }},
            "technical": {{ "score": 0, "evidence": "", "strong_signal": "", "red_flag": "" }},
            "problem_solving": {{ "score": 0, "evidence": "", "strong_signal": "", "red_flag": "" }},
            "culture_fit": {{ "score": 0, "evidence": "", "strong_signal": "", "red_flag": "" }}
        }},
        "ai_evaluation_insight": {{
            "key_strengths": ["...", "..."],
            "growth_areas": ["...", "..."]
        }},
        "machine_recommendation": "RECOMMENDED FOR HIRE / CONSIDER / REJECT",
        "executive_summary": "1 kalimat ringkasan."
    }}
    """

    try:
        resp = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict technical evaluator. Output valid JSON only."},
                {"role": "user", "content": prompt_analyzer}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        hasil_json = json.loads(resp.choices[0].message.content)

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("UPDATE ai_interview_analyzer_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("COMPLETED", datetime.utcnow(), json.dumps(hasil_json), record_id))
        conn.commit()
        cur.close()
        conn.close()
        print(f"[DB] Record {record_id} — status: COMPLETED")

    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("UPDATE ai_interview_analyzer_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("FAILED", datetime.utcnow(), json.dumps({"error": str(e)}), record_id))
        conn.commit()
        cur.close()
        conn.close()

# ==========================================
# 6. BACKGROUND: ANALYZE CV EMPLOYER
# ==========================================
def process_cv_analysis(record_id: str, application_id: str, job_id: str, job_title: str, job_industry: str, job_requirements: str, cv_bytes: bytes):
    print(f"\n[BACKGROUND] Mulai analisa CV untuk Application ID: {application_id}...")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("INSERT INTO ai_cv_analysis_result (id, application_id, job_id, status, result) VALUES (%s, %s, %s, %s, %s)",
                (record_id, application_id, job_id, "PROCESSING", None))
    conn.commit()
    cur.close()
    conn.close()
    print(f"[DB] Record {record_id} — status: PROCESSING")

    try:
        cv_text = extract_text_from_bytes(cv_bytes)

        prompt = f"""
        Lu adalah AI HR Recruiter Senior yang SANGAT KETAT, ANALITIS, tapi juga BIJAKSANA dalam melihat potensi kandidat.
        Tugas lu mengevaluasi CV kandidat untuk posisi yang sedang dibuka.

        TARGET POSISI (JOB DESCRIPTION):
        Title: {job_title}
        Industry: {job_industry or 'Tidak disebutkan'}
        Requirements: {job_requirements}

        CV KANDIDAT:
        {cv_text}

        === ATURAN EVALUASI ===
        1. FOKUS PADA KEKURANGAN & REALITA TEKNIS: Cari tau apakah kandidat benar-benar bisa bekerja sesuai JD.
        2. SKILLS OVER TITLES: Abaikan perbedaan nama jabatan masa lalu JIKA tech stack-nya relevan.
        3. SYARAT MUTLAK: Pekerjaan medis (Dokter) wajib memiliki background kedokteran/STR.
        4. TELITI (ANTI-BLINDSPOT): Baca bagian skill CV secara CASE-INSENSITIVE.
        5. KESETARAAN TOOLS: Jangan terlalu kaku pada "merk" tools. Hargai tools yang setara sebagai fondasi yang kuat.
        6. DOMAIN KNOWLEDGE: Jika kandidat pernah bekerja di industri yang mirip ({job_industry or 'industri ini'}), jadikan ini poin plus di hr_consideration.

        Return ONLY valid JSON dengan format ini:
        {{
            "match_score": 85,
            "recommendation": "Lanjut / Tidak Lanjut / Pertimbangkan",
            "ai_reason": "1 kalimat tajam menyoroti gap/kekurangan/kelebihan utama",
            "matching_skills": ["skill match 1", "skill match 2"],
            "missing_skills": ["skill yang hilang 1", "skill yang hilang 2"],
            "hr_consideration": "Saran bijak untuk HR menimbang potensi vs kekurangan teknis."
        }}
        """

        resp = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict but wise HR API evaluating a candidate. You output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        hasil = json.loads(resp.choices[0].message.content)

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("UPDATE ai_cv_analysis_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("COMPLETED", datetime.utcnow(), json.dumps(hasil), record_id))
        conn.commit()
        cur.close()
        conn.close()
        print(f"[DB] Record {record_id} — status: COMPLETED")

    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("UPDATE ai_cv_analysis_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("FAILED", datetime.utcnow(), json.dumps({"error": str(e)}), record_id))
        conn.commit()
        cur.close()
        conn.close()

# ==========================================
# 7. BACKGROUND: RECOMMEND JOBS
# ==========================================
def process_job_recommendation(record_id: str, application_id: str, seeker_name: str, cv_bytes: bytes, jobs: list):
    print(f"\n[BACKGROUND] Mulai rekomendasi lowongan untuk Application ID: {application_id}...")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("INSERT INTO ai_recommended_job_result (id, application_id, status, result) VALUES (%s, %s, %s, %s)",
                (record_id, application_id, "PROCESSING", None))
    conn.commit()
    cur.close()
    conn.close()
    print(f"[DB] Record {record_id} — status: PROCESSING")

    try:
        cv_text = extract_text_from_bytes(cv_bytes)
        jobs_payload = [{"id": i, "title": j.get("title",""), "company": j.get("company",""), "industry": j.get("industry",""), "requirements": j.get("requirements","")} for i, j in enumerate(jobs)]

        prompt = f"""
        Lu adalah AI Career Coach eksklusif untuk {seeker_name}.
        Tugas lu mencarikan peluang kerja terbaik dari DAFTAR LOWONGAN berdasarkan CV-nya, dan memberikan feedback LANGSUNG kepadanya.

        CV {seeker_name}:
        {cv_text}

        DAFTAR LOWONGAN (JSON):
        {json.dumps(jobs_payload)}

        === ATURAN REKOMENDASI KANDIDAT ===
        1. SUDUT PANDANG (POV) KANDIDAT: Bicaralah LANGSUNG kepada {seeker_name} menggunakan kata ganti "Kamu". DILARANG menggunakan sudut pandang orang ketiga.
        2. BYPASS FULLSTACK: Pengalaman "Fullstack" di CV = Lulus lowongan "Frontend" atau "Backend".
        3. ATURAN SKORING POSISI DATA (TAMENG BAJA):
           - Data Engineer / DWH: Jika TIDAK ADA skill ETL/Airflow/SSIS/PySpark, maksimal skor 30.
           - Data Analyst / BI: Jika punya SQL, Python, dan Reporting (SSRS), WAJIB berikan skor minimal 75!
        4. CARA MENULIS 'why_it_fits': WAJIB spesifik menyebutkan tools dari CV dan konteks industri perusahaan lama kandidat.
        5. GAYA BAHASA: DILARANG pakai kalimat repetitif. Buat mengalir dan persuasif!

        HANYA kembalikan lowongan dengan match_score (integer) >= 50.
        Return ONLY valid JSON dengan format ini:
        {{
            "recommendations": [
                {{
                    "job_title": "Nama posisi",
                    "company": "Nama perusahaan",
                    "industry": "Nama industri",
                    "match_score": 85,
                    "why_it_fits": "Kalimat spesifik yang menyebutkan tools dan konteks industri."
                }}
            ]
        }}
        """

        resp = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a personalized Career Coach API. You output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.4
        )
        hasil = json.loads(resp.choices[0].message.content)
        recommendations = sorted(hasil.get("recommendations", []), key=lambda x: x['match_score'], reverse=True)[:10]

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("UPDATE ai_recommended_job_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("COMPLETED", datetime.utcnow(), json.dumps(recommendations), record_id))
        conn.commit()
        cur.close()
        conn.close()
        print(f"[DB] Record {record_id} — status: COMPLETED")

    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("UPDATE ai_recommended_job_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("FAILED", datetime.utcnow(), json.dumps({"error": str(e)}), record_id))
        conn.commit()
        cur.close()
        conn.close()

# ==========================================
# 8. ENDPOINTS
# ==========================================
@app.post("/analyze-interview")
async def analyze_interview(payload: InterviewPayload, background_tasks: BackgroundTasks):
    print(f"\n[REQUEST /analyze-interview] {payload.model_dump_json(indent=2)}")

    if not payload.transcript:
        raise HTTPException(status_code=400, detail="Transcript kosong!")

    record_id = str(uuid.uuid4())
    background_tasks.add_task(process_interview_background, record_id, payload)

    return {
        "status": "success",
        "message": "Data berhasil diterima. AI sedang menganalisa di background.",
        "jobId": payload.jobId
    }


@app.post("/analyze-cv-employer")
async def analyze_cv_employer(
    background_tasks: BackgroundTasks,
    applicationId: str = Form(...),
    jobId: str = Form(...),
    jobTitle: str = Form(...),
    jobRequirements: str = Form(...),
    jobIndustry: Optional[str] = Form(None),
    cvFile: UploadFile = File(...)
):
    print(f"\n[REQUEST /analyze-cv-employer] applicationId={applicationId} jobId={jobId} jobTitle={jobTitle}")

    if not cvFile.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File harus berformat PDF.")

    cv_bytes = await cvFile.read()
    record_id = str(uuid.uuid4())
    background_tasks.add_task(process_cv_analysis, record_id, applicationId, jobId, jobTitle, jobIndustry, jobRequirements, cv_bytes)

    return {
        "status": "success",
        "message": "Data berhasil diterima. AI sedang menganalisa di background.",
        "applicationId": applicationId,
        "jobId": jobId
    }


@app.post("/recommend-jobs")
async def recommend_jobs(
    background_tasks: BackgroundTasks,
    applicationId: str = Form(...),
    seekerName: str = Form(...),
    jobs: str = Form(...),
    cvFile: UploadFile = File(...)
):
    print(f"\n[REQUEST /recommend-jobs] applicationId={applicationId} seekerName={seekerName}")

    if not cvFile.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File harus berformat PDF.")

    try:
        jobs_list = json.loads(jobs)
    except Exception:
        raise HTTPException(status_code=400, detail="Field 'jobs' harus berupa JSON array yang valid.")

    cv_bytes = await cvFile.read()
    record_id = str(uuid.uuid4())
    background_tasks.add_task(process_job_recommendation, record_id, applicationId, seekerName, cv_bytes, jobs_list)

    return {
        "status": "success",
        "message": "Data berhasil diterima. AI sedang mencari rekomendasi lowongan di background.",
        "applicationId": applicationId
    }
