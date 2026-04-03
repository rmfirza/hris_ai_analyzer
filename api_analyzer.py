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
MIN_MATCH_SCORE = 50
MAX_RECOMMENDATIONS = 10

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

app = FastAPI(title="HRIS AI Analyzer API", version="3.0")

# ==========================================
# 2. DB HELPER & SETUP TABLES
# ==========================================
def db_execute(query, params=None):
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
    finally:
        conn.close()

def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
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
    finally:
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
    applicationId: str
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

    db_execute("""
        INSERT INTO ai_interview_analyzer_result
        (id, application_id, job_id, room_id, room_name, interviewed_at, status, result)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (record_id, payload.applicationId, payload.jobId, payload.roomSid, payload.roomName, payload.endedAt, "PROCESSING", None))
    print(f"[DB] Record {record_id} — status: PROCESSING")

    conversation_history = ""
    for chat in payload.transcript:
        role = "Interviewer" if chat.role == "assistant" else "Candidate"
        conversation_history += f"{role}: {chat.text}\n"

    prompt_analyzer = f"""
    Lu adalah HRD Manager (Enterprise Level) yang SANGAT TELITI, ANTI-BIAS, dan punya insting tajam.
    Tugas lu menganalisa TRANSKRIP WAWANCARA KANDIDAT dan memberikan penilaian objektif berbasis BUKTI MUTLAK.

    TRANSKRIP WAWANCARA:
    {conversation_history}

    === RUBRIK PENILAIAN MUTLAK & GUARDRAILS ===
    1. ZERO HALLUCINATION: JIKA kandidat tidak menyebutkan suatu skill/pengalaman secara eksplisit, asumsikan mereka TIDAK BISA. Jangan menebak-nebak.
    2. BULLSHIT DETECTION (COMMUNICATION): Beri penalti (skor < 6) jika kandidat menjawab muter-muter, terlalu banyak teori tanpa contoh nyata (lack of action), atau menghindari inti pertanyaan.
    3. DEPTH OF KNOWLEDGE (TECHNICAL): Bedakan antara "Pernah pakai" vs "Paham cara kerjanya". Nilai tinggi (8-10) HANYA untuk kandidat yang bisa menjelaskan 'Mengapa' (trade-offs, scalability, arsitektur), bukan sekadar 'Bagaimana'.
    4. INDEPENDENCE (PROBLEM SOLVING): Nilai rendah jika cara debugging kandidat adalah "langsung tanya senior" tanpa ada inisiatif cek log, baca dokumentasi, atau isolasi masalah sendiri.
    5. MATURITY (CULTURE FIT): Cari sinyal kepemilikan (ownership) dan kolaborasi pragmatis (bukan memaksakan ego).

    === ATURAN FORMAT OUTPUT ===
    Wajib sertakan: "score", "evidence" (kutipan langsung 1-2 kalimat), "strong_signal", dan "red_flag".
    Jika tidak ada sinyal kuat/red flag, tulis "None".

    ATURAN "machine_recommendation":
    - Hanya boleh salah satu dari 3 nilai ini: "RECOMMENDED_FOR_HIRE", "CONSIDER", atau "REJECT".

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
        "machine_recommendation": "RECOMMENDED_FOR_HIRE | CONSIDER | REJECT",
        "executive_summary": "1 kalimat ringkasan tajam."
    }}
    """

    try:
        resp = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": "Lu adalah evaluator teknis yang ketat dan objektif. Output HANYA valid JSON, tanpa teks lain."},
                {"role": "user", "content": prompt_analyzer}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        hasil_json = json.loads(resp.choices[0].message.content)

        db_execute("UPDATE ai_interview_analyzer_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("COMPLETED", datetime.utcnow(), json.dumps(hasil_json), record_id))
        print(f"[DB] Record {record_id} — status: COMPLETED")

    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        db_execute("UPDATE ai_interview_analyzer_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("FAILED", datetime.utcnow(), json.dumps({"error": str(e)}), record_id))

# ==========================================
# 6. BACKGROUND: ANALYZE CV EMPLOYER
# ==========================================
def process_cv_analysis(record_id: str, application_id: str, job_id: str, job_title: str, job_industry: str, job_requirements: str, cv_bytes: bytes):
    print(f"\n[BACKGROUND] Mulai analisa CV untuk Application ID: {application_id}...")

    db_execute("INSERT INTO ai_cv_analysis_result (id, application_id, job_id, status, result) VALUES (%s, %s, %s, %s, %s)",
                (record_id, application_id, job_id, "PROCESSING", None))
    print(f"[DB] Record {record_id} — status: PROCESSING")

    try:
        cv_text = extract_text_from_bytes(cv_bytes)
        if not cv_text:
            raise ValueError("CV PDF tidak memiliki teks yang bisa diekstrak.")

        prompt = f"""
        Lu adalah Sistem Applicant Tracking System (ATS) Level Enterprise yang SANGAT KETAT, OBJEKTIF, dan BIJAKSANA.

        TARGET POSISI (JOB DESCRIPTION):
        Title: {job_title}
        Industry: {job_industry or 'Umum'}
        Requirements: {job_requirements}

        CV KANDIDAT:
        {cv_text}

        === ATURAN EVALUASI (BLIND HIRING & MERITOCRACY) ===
        1. BLIND HIRING PROTOCOL: Abaikan nama, umur, gender, ras, atau universitas. Fokus 100% pada SKILL dan PENGALAMAN.
        2. RECENCY WEIGHTING (BOBOT WAKTU): Skill yang digunakan di pengalaman kerja TERAKHIR memiliki bobot jauh lebih tinggi daripada skill yang digunakan 5 tahun lalu.
        3. OVERQUALIFIED CHECK: Jika CV menunjukkan pengalaman level Director/VP tapi melamar posisi Junior/Staff, berikan status "Pertimbangkan" dengan ai_reason "Kandidat berpotensi overqualified (Flight Risk)".
        4. DOMAIN KNOWLEDGE TRANSFER: Hargai pengalaman di industri yang sama. Jika beda industri, lihat apakah proses bisnisnya mirip (misal: E-commerce dan Logistik).
        5. INVALID DOCUMENT: Jika teks tidak terdeteksi sebagai CV yang wajar (misal: resep masakan, dokumen acak), berikan match_score 0 dan recommendation "Tidak Lanjut".

        ATURAN SKOR:
        - match_score adalah integer dari 0 sampai 100.
        - recommendation hanya boleh salah satu dari: "Lanjut", "Tidak Lanjut", atau "Pertimbangkan".

        Return ONLY valid JSON dengan format ini:
        {{
            "match_score": 85,
            "recommendation": "Lanjut | Tidak Lanjut | Pertimbangkan",
            "ai_reason": "1 kalimat tajam menyoroti gap/kekurangan/kelebihan utama",
            "matching_skills": ["skill match 1", "skill match 2"],
            "missing_skills": ["skill mutlak yang hilang 1", "skill mutlak yang hilang 2"],
            "hr_consideration": "Saran level-direktur untuk HR."
        }}
        """

        resp = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": "Lu adalah sistem ATS enterprise yang ketat dan objektif. Output HANYA valid JSON, tanpa teks lain."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        hasil = json.loads(resp.choices[0].message.content)

        db_execute("UPDATE ai_cv_analysis_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("COMPLETED", datetime.utcnow(), json.dumps(hasil), record_id))
        print(f"[DB] Record {record_id} — status: COMPLETED")

    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        db_execute("UPDATE ai_cv_analysis_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("FAILED", datetime.utcnow(), json.dumps({"error": str(e)}), record_id))

# ==========================================
# 7. BACKGROUND: RECOMMEND JOBS
# ==========================================
def process_job_recommendation(record_id: str, application_id: str, seeker_name: str, cv_bytes: bytes, jobs: list):
    print(f"\n[BACKGROUND] Mulai rekomendasi lowongan untuk Application ID: {application_id}...")

    db_execute("INSERT INTO ai_recommended_job_result (id, application_id, status, result) VALUES (%s, %s, %s, %s)",
                (record_id, application_id, "PROCESSING", None))
    print(f"[DB] Record {record_id} — status: PROCESSING")

    try:
        cv_text = extract_text_from_bytes(cv_bytes)
        if not cv_text:
            raise ValueError("CV PDF tidak memiliki teks yang bisa diekstrak.")

        jobs_payload = [{"id": i, "title": j.get("title",""), "company": j.get("company",""), "industry": j.get("industry",""), "requirements": j.get("requirements","")} for i, j in enumerate(jobs)]

        prompt = f"""
        Lu adalah Elite Career Strategist & Coach eksklusif untuk {seeker_name}.
        Tugas lu menganalisa CV-nya, mencarikan peluang kerja terbaik dari DAFTAR LOWONGAN, dan memberikan strategi yang ACTIONABLE.

        CV {seeker_name}:
        {cv_text}

        DAFTAR LOWONGAN (JSON):
        {json.dumps(jobs_payload)}

        === ATURAN REKOMENDASI (STRATEGIC COACHING) ===
        1. GAYA BAHASA: POV orang pertama ke orang kedua ("Saya melihat kamu..."). Inspiratif, profesional, dan to-the-point. Jangan repetitif.
        2. DYNAMIC WORLD KNOWLEDGE: Ekstrak industri perusahaan lama {seeker_name} dari CV menggunakan pengetahuan internalmu (misal: Telkomsel = Telco, BCA = Finance).
        3. BYPASS & EQUIVALENCE: Fullstack bisa masuk Frontend/Backend. SSRS setara dengan dasar Tableau/PowerBI.
        4. 'WHY IT FITS': Jelaskan korelasi spesifik antara tools di CV dengan requirement lowongan, dan bagaimana pengalaman industri lamanya membawa "unfair advantage" di industri baru ini.
        5. 'WHAT TO IMPROVE' (CRITICAL): Karena jarang ada kecocokan 100%, berikan 1 saran teknis/soft-skill paling krusial yang harus dipelajari {seeker_name} untuk menutupi gap dengan JD ini.

        ATURAN SKOR:
        - match_score adalah integer dari 0 sampai 100.
        - HANYA kembalikan lowongan dengan match_score >= {MIN_MATCH_SCORE}.

        Return ONLY valid JSON dengan format ini:
        {{
            "recommendations": [
                {{
                    "job_title": "Nama posisi",
                    "company": "Nama perusahaan",
                    "industry": "Nama industri",
                    "match_score": 85,
                    "why_it_fits": "Penjelasan mengapa pengalamannya sangat relevan.",
                    "what_to_improve": "1 hal spesifik yang menjadi gap dan harus segera dipelajari (misal: 'Pelajari AWS untuk melengkapi stack Node.js kamu')."
                }}
            ]
        }}
        """

        resp = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": "Lu adalah Elite Career Strategist yang memberikan rekomendasi kerja strategis. Output HANYA valid JSON, tanpa teks lain."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        hasil = json.loads(resp.choices[0].message.content)
        recommendations = [r for r in hasil.get("recommendations", []) if r.get("match_score", 0) >= MIN_MATCH_SCORE]
        recommendations = sorted(recommendations, key=lambda x: x['match_score'], reverse=True)[:MAX_RECOMMENDATIONS]

        db_execute("UPDATE ai_recommended_job_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("COMPLETED", datetime.utcnow(), json.dumps(recommendations), record_id))
        print(f"[DB] Record {record_id} — status: COMPLETED")

    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        db_execute("UPDATE ai_recommended_job_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("FAILED", datetime.utcnow(), json.dumps({"error": str(e)}), record_id))

# ==========================================
# 8. ENDPOINTS
# ==========================================
@app.get("/")
def health_check():
    return {"status": "ok"}


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
