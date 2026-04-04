import os
import io
import json
import uuid
import time
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

# HANYA PAKAI 1 MODEL NGEBUT SEKARANG
LLAMA_MODEL = "llama-3.3-70b-versatile"
MIN_MATCH_SCORE = 50
MAX_RECOMMENDATIONS = 10

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

app = FastAPI(title="HRIS AI Analyzer API - LLAMA SPEED", version="3.2")

# ==========================================
# 2. DB HELPER & SETUP TABLES
# ==========================================
def db_execute(query, params=None):
    # Menggunakan WITH agar koneksi dan cursor otomatis tertutup dengan aman
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            conn.commit()

def init_db():
    db_execute("""
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
    db_execute("""
        CREATE TABLE IF NOT EXISTS ai_cv_analysis_result (
            id              UUID PRIMARY KEY,
            application_id  VARCHAR(255),
            job_id          VARCHAR(255),
            analyzed_at     TIMESTAMP,
            status          VARCHAR(50),
            result          JSONB
        )
    """)
    db_execute("""
        CREATE TABLE IF NOT EXISTS ai_recommended_job_result (
            id              UUID PRIMARY KEY,
            application_id  VARCHAR(255),
            analyzed_at     TIMESTAMP,
            status          VARCHAR(50),
            result          JSONB
        )
    """)
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
    applicationId: str
    jobId: str
    roomName: Optional[str] = None
    roomSid: Optional[str] = None
    endedAt: Optional[str] = None
    transcript: List[TranscriptItem]

# ==========================================
# 5. BACKGROUND: ANALYZE INTERVIEW (MERGED LOGIC + TIMER)
# ==========================================
def process_interview_background(record_id: str, payload: InterviewPayload):
    start_time = time.time()
    print(f"\n[BACKGROUND] Mulai analisa interview Job ID: {payload.jobId}...")

    db_execute("""
        INSERT INTO ai_interview_analyzer_result
        (id, application_id, job_id, room_id, room_name, interviewed_at, status, result)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (record_id, payload.applicationId, payload.jobId, payload.roomSid, payload.roomName, payload.endedAt, "PROCESSING", None))

    conversation_history = ""
    for chat in payload.transcript:
        role = "Interviewer" if chat.role == "assistant" else "Candidate"
        conversation_history += f"{role}: {chat.text}\n"

    # MERGE LOGIC: Rubrik asli lu + Guardrails Enterprise gue
    prompt_analyzer = f"""
    Lu adalah AI Engineering Manager & Senior Technical Recruiter (Enterprise Level) yang SANGAT TELITI, ANTI-BIAS, dan punya insting tajam.
    Tugas lu menganalisa TRANSKRIP WAWANCARA KANDIDAT dan memberikan penilaian objektif beserta BUKTI (Evidence).

    TRANSKRIP WAWANCARA:
    {conversation_history}

    === RUBRIK PENILAIAN MUTLAK & GUARDRAILS ===
    Berikan skor (1-10) untuk setiap kategori dengan mematuhi aturan berikut:
    1. ZERO HALLUCINATION: Jika kandidat tidak menyebutkan skill/pengalaman secara eksplisit, asumsikan TIDAK BISA. Jangan menebak.
    2. COMMUNICATION (BULLSHIT DETECTION): Beri penalti (skor < 6) jika jawaban muter-muter, terlalu banyak teori tanpa contoh nyata, atau menghindari inti pertanyaan.
    3. TECHNICAL (DEPTH OF KNOWLEDGE): Bedakan "Pernah pakai" vs "Paham cara kerjanya". Skor 8-10 HANYA untuk kandidat yang bisa menjelaskan trade-offs, arsitektur, dan best practice.
    4. PROBLEM SOLVING (INDEPENDENCE): Nilai rendah jika cara debuggingnya adalah "langsung tanya senior" tanpa inisiatif cek log/isolasi masalah.
    5. CULTURE FIT (MATURITY): Cari sinyal ownership, teamwork pragmatis, dan cara menerima feedback. Penalti jika egois.

    === ATURAN FORMAT OUTPUT ===
    Wajib sertakan: "score", "evidence" (kutipan langsung 1-2 kalimat), "strong_signal", dan "red_flag".
    Jika tidak ada sinyal kuat/red flag, tulis "None".

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
        "machine_recommendation": "RECOMMENDED FOR HIRE | CONSIDER | REJECT",
        "executive_summary": "1 kalimat ringkasan tajam."
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
        
        duration = round(time.time() - start_time, 2)
        hasil_json["ai_processing_time_seconds"] = duration
        print(f"[TIMING] Interview selesai dalam {duration} detik.")

        db_execute("UPDATE ai_interview_analyzer_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("COMPLETED", datetime.utcnow(), json.dumps(hasil_json), record_id))

    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        db_execute("UPDATE ai_interview_analyzer_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("FAILED", datetime.utcnow(), json.dumps({"error": str(e)}), record_id))

# ==========================================
# 6. BACKGROUND: ANALYZE CV EMPLOYER (MERGED LOGIC + TIMER)
# ==========================================
def process_cv_analysis(record_id: str, application_id: str, job_id: str, job_title: str, job_industry: str, job_requirements: str, cv_bytes: bytes):
    start_time = time.time()
    print(f"\n[BACKGROUND] Mulai analisa CV App ID: {application_id}...")

    db_execute("INSERT INTO ai_cv_analysis_result (id, application_id, job_id, status, result) VALUES (%s, %s, %s, %s, %s)",
                (record_id, application_id, job_id, "PROCESSING", None))

    try:
        cv_text = extract_text_from_bytes(cv_bytes)
        if len(cv_text) < 20: 
            raise ValueError("CV PDF kosong atau bukan teks yang valid.")

        # MERGE LOGIC: Rule praktis lu + Rule enterprise gue
        prompt = f"""
        Lu adalah AI HR Recruiter Senior & Sistem ATS Enterprise yang SANGAT KETAT, ANALITIS, tapi juga BIJAKSANA.
        Tugas lu mengevaluasi CV kandidat untuk posisi yang sedang dibuka.

        TARGET POSISI (JOB DESCRIPTION):
        Title: {job_title}
        Industry: {job_industry or 'Tidak disebutkan'}
        Requirements: {job_requirements}

        CV KANDIDAT:
        {cv_text}

        === ATURAN EVALUASI (BLIND HIRING & MERITOCRACY) ===
        1. FOKUS PADA REALITA TEKNIS & KEKURANGAN: Cari tau apakah kandidat benar-benar bisa bekerja sesuai JD.
        2. SKILLS OVER TITLES: Abaikan perbedaan nama jabatan masa lalu JIKA tech stack-nya relevan.
        3. SYARAT MUTLAK: Pekerjaan dengan sertifikasi wajib (misal medis/hukum) harus dicek ketat.
        4. KESETARAAN TOOLS: Jangan kaku pada "merk". Hargai tools yang setara (misal MySQL vs PostgreSQL) sebagai fondasi kuat.
        5. BLIND HIRING: Abaikan gender, umur, ras, atau status pernikahan. Fokus pada skill.
        6. RECENCY WEIGHTING: Skill yang dipakai di pekerjaan TERAKHIR bobotnya jauh lebih tinggi dari skill 5 tahun lalu.
        7. OVERQUALIFIED CHECK: Jika CV level Director melamar posisi Junior, beri status "Pertimbangkan" dengan ai_reason "Berpotensi overqualified (Flight Risk)".
        8. DOMAIN KNOWLEDGE: Jika pernah bekerja di industri yang mirip ({job_industry or 'industri ini'}), jadikan poin plus besar di hr_consideration.

        Return ONLY valid JSON dengan format ini:
        {{
            "match_score": 85,
            "recommendation": "Lanjut | Tidak Lanjut | Pertimbangkan",
            "ai_reason": "1 kalimat tajam menyoroti gap/kekurangan/kelebihan utama",
            "matching_skills": ["skill match 1", "skill match 2"],
            "missing_skills": ["skill mutlak yang hilang 1", "skill mutlak yang hilang 2"],
            "hr_consideration": "Saran level-direktur untuk HR menimbang potensi vs kekurangan teknis."
        }}
        """

        resp = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict but wise HR ATS API evaluating a candidate. You output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        hasil = json.loads(resp.choices[0].message.content)
        
        duration = round(time.time() - start_time, 2)
        hasil["ai_processing_time_seconds"] = duration
        print(f"[TIMING] CV selesai dalam {duration} detik.")

        db_execute("UPDATE ai_cv_analysis_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("COMPLETED", datetime.utcnow(), json.dumps(hasil), record_id))

    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        db_execute("UPDATE ai_cv_analysis_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("FAILED", datetime.utcnow(), json.dumps({"error": str(e)}), record_id))

# ==========================================
# 7. BACKGROUND: RECOMMEND JOBS (ORIGINAL FLAVOR + TIMER)
# ==========================================
def process_job_recommendation(record_id: str, application_id: str, seeker_name: str, cv_bytes: bytes, jobs: list):
    start_time = time.time()
    print(f"\n[BACKGROUND] Mulai rekomendasi lowongan App ID: {application_id}...")

    db_execute("DELETE FROM ai_recommended_job_result WHERE application_id = %s", (application_id,))
    db_execute("INSERT INTO ai_recommended_job_result (id, application_id, status, result) VALUES (%s, %s, %s, %s)",
                (record_id, application_id, "PROCESSING", None))

    try:
        cv_text = extract_text_from_bytes(cv_bytes)
        jobs_payload = [{"id": i, "title": j.get("title",""), "company": j.get("company",""), "industry": j.get("industry",""), "requirements": j.get("requirements","")} for i, j in enumerate(jobs)]

        # --- MENGGUNAKAN PROMPT ASLI LU YANG TERBUKTI AMPUH ---
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
        4. CARA MENULIS 'why_it_fits': WAJIB spesifik menyebutkan tools dari CV dan bandingkan industri perusahaan lama kandidat dengan industri perusahaan lowongan baru.
        5. GAYA BAHASA: DILARANG pakai kalimat repetitif. Buat mengalir, inspiratif, dan persuasif!
        6. WHAT TO IMPROVE: Berikan 1 saran teknis spesifik untuk dipelajari guna menutupi requirement yang kurang.

        HANYA kembalikan lowongan dengan match_score (integer) >= {MIN_MATCH_SCORE}.
        Return ONLY valid JSON dengan format ini:
        {{
            "recommendations": [
                {{
                    "job_title": "Nama posisi",
                    "company": "Nama perusahaan",
                    "industry": "Nama industri",
                    "match_score": 85,
                    "why_it_fits": "Kalimat spesifik yang mengaitkan tools CV dan konteks industri lama dengan industri baru.",
                    "what_to_improve": "1 hal spesifik yang menjadi gap dan harus dipelajari."
                }}
            ]
        }}
        """

        resp = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a personalized Career Coach API. You output valid JSON only and never repeat sentence structures."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.4 # Suhu 0.4 biar Llama lumayan kreatif nulis why_it_fits-nya
        )
        
        hasil = json.loads(resp.choices[0].message.content)
        
        # Urutkan berdasarkan skor tertinggi dan ambil maksimal 10 rekomendasi
        recommendations = sorted(hasil.get("recommendations", []), key=lambda x: x['match_score'], reverse=True)[:MAX_RECOMMENDATIONS]
        
        duration = round(time.time() - start_time, 2)
        
        # Bungkus hasil akhirnya biar rapi
        final_result = {
            "ai_processing_time_seconds": duration,
            "total_recommended": len(recommendations),
            "recommendations": recommendations
        }
        
        print(f"[TIMING] Rekomendasi selesai dalam {duration} detik.")

        db_execute("UPDATE ai_recommended_job_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("COMPLETED", datetime.utcnow(), json.dumps(final_result), record_id))

    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        db_execute("UPDATE ai_recommended_job_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("FAILED", datetime.utcnow(), json.dumps({"error": str(e)}), record_id))

# ==========================================
# 8. POST ENDPOINTS
# ==========================================
@app.get("/")
def health_check():
    return {"status": "ok, LLAMA 3.3 Engine Online!"}

@app.post("/analyze-interview")
async def analyze_interview(payload: InterviewPayload, background_tasks: BackgroundTasks):
    if not payload.transcript: raise HTTPException(status_code=400, detail="Transcript kosong!")
    record_id = str(uuid.uuid4())
    background_tasks.add_task(process_interview_background, record_id, payload)
    return {"status": "success", "message": "Diproses Llama AI di background.", "applicationId": payload.applicationId, "jobId": payload.jobId}

@app.post("/analyze-cv-employer")
async def analyze_cv_employer(
    background_tasks: BackgroundTasks, applicationId: str = Form(...), jobId: str = Form(...), jobTitle: str = Form(...),
    jobRequirements: str = Form(...), jobIndustry: Optional[str] = Form(None), cvFile: UploadFile = File(...)
):
    if not cvFile.filename.endswith(".pdf"): raise HTTPException(status_code=400, detail="Harus PDF.")
    cv_bytes = await cvFile.read()
    record_id = str(uuid.uuid4())
    background_tasks.add_task(process_cv_analysis, record_id, applicationId, jobId, jobTitle, jobIndustry, jobRequirements, cv_bytes)
    return {"status": "success", "message": "Diproses Llama AI di background.", "applicationId": applicationId, "jobId": jobId}

@app.post("/recommend-jobs")
async def recommend_jobs(
    background_tasks: BackgroundTasks, applicationId: str = Form(...), seekerName: str = Form(...),
    jobs: str = Form(...), cvFile: UploadFile = File(...)
):
    if not cvFile.filename.endswith(".pdf"): raise HTTPException(status_code=400, detail="Harus PDF.")
    try: jobs_list = json.loads(jobs)
    except Exception: raise HTTPException(status_code=400, detail="Field jobs tidak valid.")
    cv_bytes = await cvFile.read()
    record_id = str(uuid.uuid4())
    background_tasks.add_task(process_job_recommendation, record_id, applicationId, seekerName, cv_bytes, jobs_list)
    return {"status": "success", "message": "Diproses Llama AI di background.", "applicationId": applicationId}

# ==========================================
# 9. GET ENDPOINTS (Berbasis applicationId)
# ==========================================
@app.get("/result/interview/{application_id}")
async def get_interview_result(application_id: str):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status, result FROM ai_interview_analyzer_result WHERE application_id = %s ORDER BY analyzed_at DESC LIMIT 1", (application_id,))
            row = cur.fetchone()
    if not row: raise HTTPException(status_code=404, detail="Data tidak ditemukan")
    return {"application_id": application_id, "status": row[0], "data": row[1]}

@app.get("/result/cv-employer/{application_id}")
async def get_cv_employer_result(application_id: str):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status, result FROM ai_cv_analysis_result WHERE application_id = %s ORDER BY analyzed_at DESC LIMIT 1", (application_id,))
            row = cur.fetchone()
    if not row: raise HTTPException(status_code=404, detail="Data tidak ditemukan")
    return {"application_id": application_id, "status": row[0], "data": row[1]}

@app.get("/result/recommend-jobs/{application_id}")
async def get_recommend_jobs_result(application_id: str):
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status, result FROM ai_recommended_job_result WHERE application_id = %s ORDER BY analyzed_at DESC LIMIT 1", (application_id,))
            row = cur.fetchone()
    if not row: raise HTTPException(status_code=404, detail="Data tidak ditemukan")
    return {"application_id": application_id, "status": row[0], "data": row[1]}