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

TEXT_MODEL = "llama-3.3-70b-versatile"
REASONING_MODEL = "openai/gpt-oss-120b"
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
# 5. BACKGROUND: ANALYZE INTERVIEW (2-STEP)
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

    try:
        # --- STEP 1: TEXT MODEL — Ekstraksi & kategorisasi transkrip ---
        step1_prompt = f"""
        Analisa transkrip wawancara berikut dan ekstrak informasi secara OBJEKTIF tanpa menambahkan asumsi.

        TRANSKRIP:
        {conversation_history}

        Ekstrak ke dalam 4 kategori berikut:
        1. COMMUNICATION: Apakah kandidat menjawab langsung ke inti pertanyaan? Atau muter-muter? Kutip 1-2 kalimat bukti.
        2. TECHNICAL: Skill/teknologi apa saja yang EKSPLISIT disebutkan kandidat? Apakah penjelasannya level permukaan ("pernah pakai") atau mendalam ("paham cara kerjanya", trade-offs, arsitektur)?
        3. PROBLEM SOLVING: Bagaimana pendekatan kandidat saat menghadapi masalah? Apakah ada inisiatif mandiri (cek log, baca docs, isolasi masalah) atau langsung tanya orang lain?
        4. CULTURE FIT: Apakah ada sinyal ownership, kolaborasi, atau justru ego? Kutip buktinya.

        Return ONLY valid JSON:
        {{
            "communication": {{ "direct_answers": ["kutipan..."], "vague_answers": ["kutipan..."], "notes": "..." }},
            "technical": {{ "skills_mentioned": ["..."], "deep_understanding": ["..."], "surface_only": ["..."], "notes": "..." }},
            "problem_solving": {{ "independent_signals": ["..."], "dependent_signals": ["..."], "notes": "..." }},
            "culture_fit": {{ "ownership_signals": ["..."], "collaboration_signals": ["..."], "ego_signals": ["..."], "notes": "..." }}
        }}
        """

        resp1 = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "Lu adalah ekstraktor data dari transkrip wawancara. Ekstrak HANYA fakta yang ada, JANGAN menambahkan asumsi. Output valid JSON."},
                {"role": "user", "content": step1_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        extracted = resp1.choices[0].message.content
        print(f"[STEP 1] Ekstraksi transkrip selesai.")

        # --- STEP 2: REASONING MODEL — Scoring & judgment ---
        step2_prompt = f"""
        Lu adalah HRD Manager (Enterprise Level) yang SANGAT TELITI, ANTI-BIAS, dan punya insting tajam.
        Berikut adalah HASIL EKSTRAKSI dari transkrip wawancara kandidat yang sudah dianalisa oleh sistem:

        {extracted}

        === RUBRIK PENILAIAN & GUARDRAILS ===
        1. ZERO HALLUCINATION: Hanya nilai berdasarkan data ekstraksi di atas. JANGAN menambahkan skill/pengalaman yang tidak ada.
        2. BULLSHIT DETECTION (COMMUNICATION): Skor < 6 jika banyak vague_answers, muter-muter, atau lack of action.
        3. DEPTH OF KNOWLEDGE (TECHNICAL): Skor 8-10 HANYA jika ada deep_understanding. Jika hanya surface_only, maksimal 6.
        4. INDEPENDENCE (PROBLEM SOLVING): Skor rendah jika dependent_signals dominan tanpa independent_signals.
        5. MATURITY (CULTURE FIT): Cari ownership_signals dan collaboration_signals. Penalti jika ada ego_signals.

        Berikan skor 1-10 per kategori.
        Wajib sertakan "evidence" (kutipan langsung 1-2 kalimat), "strong_signal", dan "red_flag".
        Jika tidak ada sinyal kuat/red flag, tulis "None".

        ATURAN "machine_recommendation":
        - Hanya boleh salah satu dari 3 nilai ini: "RECOMMENDED_FOR_HIRE", "CONSIDER", atau "REJECT".

        Return ONLY valid JSON:
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

        resp2 = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": "Lu adalah HRD Manager enterprise yang ketat dan objektif. Scoring berdasarkan DATA SAJA. Output valid JSON."},
                {"role": "user", "content": step2_prompt}
            ],
            response_format={"type": "json_object"},
            reasoning_effort="high",
            temperature=0.1
        )
        hasil_json = json.loads(resp2.choices[0].message.content)
        print(f"[STEP 2] Reasoning & scoring selesai.")

        db_execute("UPDATE ai_interview_analyzer_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("COMPLETED", datetime.utcnow(), json.dumps(hasil_json), record_id))
        print(f"[DB] Record {record_id} — status: COMPLETED")

    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        db_execute("UPDATE ai_interview_analyzer_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("FAILED", datetime.utcnow(), json.dumps({"error": str(e)}), record_id))

# ==========================================
# 6. BACKGROUND: ANALYZE CV EMPLOYER (2-STEP)
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

        # --- STEP 1: TEXT MODEL — Matching CV vs JD ---
        step1_prompt = f"""
        Bandingkan CV kandidat dengan Job Description berikut secara OBJEKTIF.

        TARGET POSISI:
        Title: {job_title}
        Industry: {job_industry or 'Umum'}
        Requirements: {job_requirements}

        CV KANDIDAT:
        {cv_text}

        Ekstrak:
        1. Skills di CV yang COCOK dengan requirements JD (case-insensitive, termasuk tools setara).
        2. Skills di JD yang TIDAK ADA di CV.
        3. Pengalaman kerja terakhir kandidat (posisi, perusahaan, industri).
        4. Total tahun pengalaman relevan.
        5. Apakah dokumen ini valid CV? (true/false)

        Return ONLY valid JSON:
        {{
            "is_valid_cv": true,
            "matching_skills": ["skill1", "skill2"],
            "missing_skills": ["skill1", "skill2"],
            "latest_experience": {{ "position": "...", "company": "...", "industry": "..." }},
            "years_relevant_experience": 0,
            "notes": "Catatan tambahan jika ada."
        }}
        """

        resp1 = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "Lu adalah sistem ATS yang mengekstrak dan mencocokkan data CV vs JD. Output valid JSON."},
                {"role": "user", "content": step1_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        extracted = resp1.choices[0].message.content
        print(f"[STEP 1] Matching CV vs JD selesai.")

        # --- STEP 2: REASONING MODEL — Scoring & evaluasi ---
        step2_prompt = f"""
        Lu adalah Sistem ATS (Enterprise Level) yang KETAT, OBJEKTIF, dan BIJAKSANA.
        Berikut HASIL MATCHING CV vs JD yang sudah diekstrak oleh sistem:

        {extracted}

        KONTEKS POSISI:
        Title: {job_title}
        Industry: {job_industry or 'Umum'}

        === ATURAN EVALUASI ===
        1. BLIND HIRING: Abaikan nama, umur, gender, ras, universitas. Fokus 100% pada SKILL dan PENGALAMAN.
        2. RECENCY WEIGHTING: Skill di pengalaman terakhir bobotnya lebih tinggi.
        3. OVERQUALIFIED CHECK: Jika pengalaman level Director/VP tapi melamar Junior/Staff, status "Pertimbangkan" dengan ai_reason "Kandidat berpotensi overqualified (Flight Risk)".
        4. DOMAIN KNOWLEDGE: Hargai pengalaman industri yang sama atau proses bisnis mirip.
        5. INVALID DOCUMENT: Jika is_valid_cv = false, berikan match_score 0 dan recommendation "Tidak Lanjut".

        ATURAN SKOR:
        - match_score: integer 0-100.
        - recommendation: hanya "Lanjut", "Tidak Lanjut", atau "Pertimbangkan".

        Return ONLY valid JSON:
        {{
            "match_score": 85,
            "recommendation": "Lanjut | Tidak Lanjut | Pertimbangkan",
            "ai_reason": "1 kalimat tajam menyoroti gap/kekurangan/kelebihan utama",
            "matching_skills": ["skill match 1", "skill match 2"],
            "missing_skills": ["skill mutlak yang hilang 1", "skill mutlak yang hilang 2"],
            "hr_consideration": "Saran level-direktur untuk HR."
        }}
        """

        resp2 = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": "Lu adalah sistem ATS enterprise. Evaluasi berdasarkan DATA MATCHING saja. Output valid JSON."},
                {"role": "user", "content": step2_prompt}
            ],
            response_format={"type": "json_object"},
            reasoning_effort="high",
            temperature=0.0
        )
        hasil = json.loads(resp2.choices[0].message.content)
        print(f"[STEP 2] Reasoning & scoring selesai.")

        db_execute("UPDATE ai_cv_analysis_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("COMPLETED", datetime.utcnow(), json.dumps(hasil), record_id))
        print(f"[DB] Record {record_id} — status: COMPLETED")

    except Exception as e:
        print(f"[BACKGROUND ERROR] {e}")
        db_execute("UPDATE ai_cv_analysis_result SET status=%s, analyzed_at=%s, result=%s WHERE id=%s",
                    ("FAILED", datetime.utcnow(), json.dumps({"error": str(e)}), record_id))

# ==========================================
# 7. BACKGROUND: RECOMMEND JOBS (2-STEP)
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

        # --- STEP 1: TEXT MODEL — Matching CV vs semua lowongan ---
        step1_prompt = f"""
        Bandingkan CV kandidat dengan SETIAP lowongan di daftar berikut. Ekstrak kecocokan secara OBJEKTIF.

        CV KANDIDAT ({seeker_name}):
        {cv_text}

        DAFTAR LOWONGAN:
        {json.dumps(jobs_payload)}

        Untuk SETIAP lowongan, ekstrak:
        1. Skills di CV yang cocok dengan requirements lowongan.
        2. Skills di requirements yang tidak ada di CV (gap).
        3. Apakah industri/domain pengalaman kandidat relevan dengan lowongan ini.

        Juga ekstrak profil umum kandidat:
        - Skills utama dari CV.
        - Pengalaman terakhir (posisi, perusahaan, industri).

        Return ONLY valid JSON:
        {{
            "candidate_profile": {{
                "key_skills": ["..."],
                "latest_position": "...",
                "latest_company": "...",
                "latest_industry": "..."
            }},
            "job_matches": [
                {{
                    "id": 0,
                    "job_title": "...",
                    "company": "...",
                    "industry": "...",
                    "matched_skills": ["..."],
                    "gap_skills": ["..."],
                    "domain_relevant": true
                }}
            ]
        }}
        """

        resp1 = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "Lu adalah sistem matching CV vs lowongan kerja. Ekstrak kecocokan secara objektif. Output valid JSON."},
                {"role": "user", "content": step1_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        extracted = resp1.choices[0].message.content
        print(f"[STEP 1] Matching CV vs lowongan selesai.")

        # --- STEP 2: REASONING MODEL — Scoring & strategic coaching ---
        step2_prompt = f"""
        Lu adalah Elite Career Strategist & Coach eksklusif untuk {seeker_name}.
        Berikut HASIL MATCHING CV vs daftar lowongan yang sudah diekstrak oleh sistem:

        {extracted}

        === ATURAN REKOMENDASI (STRATEGIC COACHING) ===
        1. GAYA BAHASA: POV orang pertama ke orang kedua ("Saya melihat kamu..."). Inspiratif, profesional, dan to-the-point. Jangan repetitif.
        2. BYPASS & EQUIVALENCE: Fullstack bisa masuk Frontend/Backend. SSRS setara dengan dasar Tableau/PowerBI.
        3. 'WHY IT FITS': Jelaskan korelasi spesifik antara matched_skills dengan requirement, dan bagaimana domain_relevant membawa "unfair advantage".
        4. 'WHAT TO IMPROVE': Berdasarkan gap_skills, berikan 1 saran paling krusial yang harus dipelajari {seeker_name}.
        5. SCORING: Pertimbangkan jumlah matched vs gap skills, dan domain relevance.

        ATURAN SKOR:
        - match_score: integer 0-100.
        - HANYA kembalikan lowongan dengan match_score >= {MIN_MATCH_SCORE}.

        Return ONLY valid JSON:
        {{
            "recommendations": [
                {{
                    "job_title": "Nama posisi",
                    "company": "Nama perusahaan",
                    "industry": "Nama industri",
                    "match_score": 85,
                    "why_it_fits": "Penjelasan mengapa pengalamannya sangat relevan.",
                    "what_to_improve": "1 hal spesifik yang menjadi gap dan harus segera dipelajari."
                }}
            ]
        }}
        """

        resp2 = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": "Lu adalah Elite Career Strategist. Buat rekomendasi berdasarkan DATA MATCHING saja. Output valid JSON."},
                {"role": "user", "content": step2_prompt}
            ],
            response_format={"type": "json_object"},
            reasoning_effort="high",
            temperature=0.2
        )
        hasil = json.loads(resp2.choices[0].message.content)
        recommendations = [r for r in hasil.get("recommendations", []) if r.get("match_score", 0) >= MIN_MATCH_SCORE]
        recommendations = sorted(recommendations, key=lambda x: x['match_score'], reverse=True)[:MAX_RECOMMENDATIONS]
        print(f"[STEP 2] Reasoning & scoring selesai.")

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
