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
# 5. BACKGROUND: ANALYZE INTERVIEW (WITH MBTI, DISC & QUESTION CONTEXT)
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

    # PROMPT DI-UPGRADE DENGAN PSIKOLOGI & KONTEKS PERTANYAAN
    prompt_analyzer = f"""
    Lu adalah AI Engineering Manager, Senior Technical Recruiter, & Psikolog Perilaku (Enterprise Level) yang SANGAT TELITI dan ANTI-BIAS.
    Tugas lu menganalisa TRANSKRIP WAWANCARA KANDIDAT dan memberikan penilaian objektif, bukti kontekstual, serta profil psikologis kandidat.

    TRANSKRIP WAWANCARA:
    {conversation_history}

    === RUBRIK PENILAIAN MUTLAK & GUARDRAILS ===
    1. ZERO HALLUCINATION: Jika kandidat tidak menyebutkan skill/pengalaman secara eksplisit, asumsikan TIDAK BISA.
    2. COMMUNICATION: Beri penalti (skor < 6) jika jawaban muter-muter, terlalu banyak teori tanpa contoh nyata.
    3. TECHNICAL: Bedakan "Pernah pakai" vs "Paham cara kerjanya". Skor 8-10 HANYA untuk kandidat yang bisa menjelaskan trade-offs dan arsitektur.
    4. PROBLEM SOLVING: Nilai rendah jika cara debuggingnya adalah "langsung tanya senior/DBA" tanpa inisiatif isolasi masalah.
    5. CULTURE FIT: Cari sinyal ownership, teamwork pragmatis. Penalti keras jika egois atau menyalahkan tim lain (QA/Analyst).
    6. KONTEKS EVIDENCE (WAJIB): Setiap bukti kutipan jawaban kandidat ("evidence_quote") WAJIB didampingi oleh pertanyaan spesifik yang dilontarkan oleh Interviewer ("question_asked"). Jangan pisahkan konteksnya.
    7. PSYCHOLOGICAL PROFILING: Berdasarkan cara kandidat menjawab, mengambil keputusan, merespon masalah, dan berinteraksi dengan tim, lakukan estimasi profil MBTI dan DISC kandidat.

    === ATURAN FORMAT OUTPUT ===
    Return ONLY valid JSON dengan format ini:
    {{
        "score_breakdown": {{
            "communication": {{ "score": 0, "question_asked": "Kutipan pertanyaan interviewer", "evidence_quote": "Kutipan jawaban kandidat", "strong_signal": "", "red_flag": "" }},
            "technical": {{ "score": 0, "question_asked": "Kutipan pertanyaan interviewer", "evidence_quote": "Kutipan jawaban kandidat", "strong_signal": "", "red_flag": "" }},
            "problem_solving": {{ "score": 0, "question_asked": "Kutipan pertanyaan interviewer", "evidence_quote": "Kutipan jawaban kandidat", "strong_signal": "", "red_flag": "" }},
            "culture_fit": {{ "score": 0, "question_asked": "Kutipan pertanyaan interviewer", "evidence_quote": "Kutipan jawaban kandidat", "strong_signal": "", "red_flag": "" }}
        }},
        "psychological_profile": {{
            "personality_summary": "1-2 kalimat ringkasan kepribadian berdasarkan gaya bahasa di transkrip.",
            "estimated_mbti": "Contoh: INTJ",
            "mbti_reasoning": "Alasan singkat mengapa MBTI ini cocok berdasarkan transkrip.",
            "estimated_disc": "Contoh: High D, Low S",
            "disc_reasoning": "Alasan singkat mengapa DISC ini cocok."
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
                {"role": "system", "content": "You are a strict technical evaluator and behavioral psychologist. Output valid JSON only."},
                {"role": "user", "content": prompt_analyzer}
            ],
            response_format={"type": "json_object"},
            temperature=0.2 # Naikin suhu dikit ke 0.2 biar dia bisa nganalisa psikologisnya lebih luwes
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

        === ATURAN EVALUASI & SKORING MUTLAK (TAMENG BAJA) ===
        1. PENALTI CORE DOMAIN (SANGAT PENTING): Jika lowongan membutuhkan spesialisasi inti (misal: Data Engineer butuh Kafka/Airflow/Spark/ETL), dan kandidat HANYA memiliki skill general (Python/SQL/Golang) dari background Backend/Fullstack, BERIKAN SKOR MAKSIMAL 30! Jangan tertipu oleh kecocokan bahasa pemrograman dasar jika ekosistem arsitektur intinya tidak dikuasai.
        2. PROPORSI REQUIREMENT: Jangan memberikan skor tinggi (>= 50) hanya karena ada 1 atau 2 keyword yang cocok. Evaluasi gambaran besar. Jika alat tempur utama (core tools) dari posisi tersebut tidak ada di CV, hancurkan skornya ke bawah 40.
        3. SKILLS OVER TITLES: Abaikan perbedaan nama jabatan masa lalu JIKA tech stack intinya benar-benar relevan dan terpenuhi secara proporsional.
        4. KESETARAAN TOOLS: Jangan kaku pada "merk". Hargai tools yang setara (misal MySQL vs PostgreSQL) sebagai fondasi kuat.
        5. BLIND HIRING: Abaikan gender, umur, ras. Fokus pada skill teknis mutlak.
        6. RECENCY WEIGHTING: Skill yang dipakai di pekerjaan TERAKHIR bobotnya jauh lebih tinggi.
        7. DOMAIN KNOWLEDGE: Jika pernah bekerja di industri yang mirip ({job_industry or 'industri ini'}), jadikan poin plus besar di hr_consideration.

        Return ONLY valid JSON dengan format ini:
        {{
            "match_score": 85,
            "recommendation": "Lanjut | Tidak Lanjut | Pertimbangkan",
            "ai_reason": "1 kalimat tajam menyoroti gap mutlak yang membuat kandidat tidak cocok, atau kelebihan utamanya.",
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

        === ATURAN SKORING HOLISTIK & PROPORSI SKILL ===
        1. JANGAN TERPAKU PADA TITLE: Abaikan perbedaan nama jabatan masa lalu. Seorang Fullstack atau Backend bisa saja cocok untuk lowongan Data Engineer asalkan tech-stack dan pemahaman sistem di CV-nya benar-benar memenuhi mayoritas requirement.
        2. EVALUASI PROPORSI (GAMBARAN BESAR): Ini SANGAT PENTING. Jangan memberikan skor tinggi (>= 50) hanya karena kebetulan cocok di 1 atau 2 tools dasar (seperti sama-sama pakai SQL atau Python). Lu harus membandingkan KESELURUHAN requirement. Jika lowongan meminta 5 core skills spesifik (misal: ETL, Airflow, Spark, Python, BigQuery) dan CV hanya memiliki 2, maka skor maksimal adalah 40 karena mayoritas ekosistemnya tidak terpenuhi.
        3. TRANSFERABLE SKILLS: Jika industri perusahaan lamanya relevan dengan proses bisnis perusahaan baru, jadikan ini faktor penambah skor, tapi tetap utamakan kecocokan proporsi teknis.

        === CARA MENULIS FEEDBACK ===
        1. POV: Bicaralah LANGSUNG menggunakan "Kamu".
        2. DILARANG TEMPLATE KLISE: Buat kalimat yang mengalir dan natural.
        3. 'why_it_fits': Jelaskan secara komprehensif. Sebut nama perusahaan lamanya, sebut tools dari CV yang match dengan JD, dan jelaskan BAGAIMANA hal tersebut relevan secara gambaran besar. (Contoh: "Pengalamanmu di PT XYZ dalam merancang database dengan PostgreSQL memberikanmu fondasi arsitektur yang kuat untuk beradaptasi dengan kebutuhan data logistik di perusahaan ini.")
        4. 'what_to_improve': Berikan 1 saran teknis paling krusial untuk menutupi gap terbesar dari requirement yang belum dia miliki.

        HANYA kembalikan lowongan dengan match_score (integer) >= {MIN_MATCH_SCORE}.
        Return ONLY valid JSON dengan format ini:
        {{
            "recommendations": [
                {{
                    "job_title": "Nama posisi",
                    "company": "Nama perusahaan",
                    "industry": "Nama industri",
                    "match_score": 85,
                    "why_it_fits": "Kalimat spesifik (sebut PT lama, tools, dan konteks gambaran besar).",
                    "what_to_improve": "1 hal teknis spesifik."
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