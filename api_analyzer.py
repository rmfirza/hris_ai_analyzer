import os
import json
import uuid
import datetime
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
from dotenv import load_dotenv

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

app = FastAPI(title="HR Interview Analyzer API", version="2.0")

# ==========================================
# 2. SETUP TABLE (JALAN SAAT SERVER START)
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
            interviewed_at  TIMESTAMP,
            analyzed_at     TIMESTAMP,
            result          JSONB
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[DB] Table ai_interview_analyzer_result siap.")

init_db()

# ==========================================
# 3. DEFINISI SKEMA JSON
# ==========================================
class TranscriptItem(BaseModel):
    role: str
    text: str
    createdAt: int

class InterviewPayload(BaseModel):
    ApplicationId: Optional[str] = None
    jobId: Optional[str] = None
    roomName: Optional[str] = None
    roomSid: Optional[str] = None
    endedAt: Optional[str] = None
    transcript: List[TranscriptItem]

# ==========================================
# 4. HELPER: SIMPAN HASIL KE POSTGRESQL
# ==========================================
def save_result(payload: InterviewPayload, hasil: dict):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ai_interview_analyzer_result (id, application_id, job_id, room_id, room_name, interviewed_at, analyzed_at, result)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        str(uuid.uuid4()),
        payload.ApplicationId,
        payload.jobId,
        payload.roomSid,
        payload.roomName,
        payload.endedAt,
        datetime.datetime.utcnow(),
        json.dumps(hasil)
    ))
    conn.commit()
    cur.close()
    conn.close()
    print(f"[DB] Hasil analisa disimpan untuk Room: {payload.roomName}")

# ==========================================
# 5. BACKGROUND TASK: PROSES & SIMPAN
# ==========================================
def proses_dan_simpan(payload: InterviewPayload):
    print(f"\n[BACKGROUND] Mulai analisa untuk Room: {payload.roomName}")

    conversation_history = ""
    for chat in payload.transcript:
        role = "Interviewer" if chat.role == "assistant" else "Candidate"
        conversation_history += f"{role}: {chat.text}\n"

    prompt_analyzer = f"""
    Lu adalah AI Engineering Manager & HR Evaluator Senior yang SANGAT TELITI dan ANTI-BIAS.
    Tugas lu menganalisa TRANSKRIP WAWANCARA KANDIDAT dan memberikan penilaian objektif beserta BUKTI (Evidence).

    TRANSKRIP WAWANCARA:
    {conversation_history}

    === RUBRIK PENILAIAN MUTLAK ===
    Berikan skor (1-10) untuk setiap kategori berdasarkan indikator berikut:

    1. COMMUNICATION:
       - Kemampuan menjelaskan ide terstruktur, menjawab langsung (tidak berputar), kejelasan bahasa, dan kemampuan mendengarkan.
    2. TECHNICAL:
       - Pemahaman konsep fundamental, penjelasan keputusan teknis (mengapa memilih solusi A), menyebutkan complexity/scalability, dan best practice.
    3. PROBLEM SOLVING:
       - Pendekatan sistematis, step-by-step reasoning, ketahanan hadapi kesulitan, kemampuan evaluasi/debugging solusi sendiri secara mandiri.
    4. CULTURE FIT:
       - Sikap pragmatis vs egois, teamwork, cara menghadapi konflik/perbedaan pendapat, ownership, dan integritas.

    === ATURAN FORMAT OUTPUT ===
    Untuk setiap kategori di atas, selain memberikan skor, lu WAJIB menyertakan:
    - "evidence": Kutipan/contoh jawaban langsung dari kandidat yang mendasari skor lu.
    - "strong_signal": Nilai plus/indikator kuat (jika ada). Jika tidak ada, isi "None".
    - "red_flag": Nilai minus/bias/hal yang patut diwaspadai (jika ada). Jika tidak ada, isi "None".

    Return ONLY valid JSON dengan format persis seperti ini:
    {{
        "score_breakdown": {{
            "communication": {{
                "score": 8,
                "evidence": "Kandidat menjawab...",
                "strong_signal": "Artikulasi sangat jelas",
                "red_flag": "None"
            }},
            "technical": {{
                "score": 7,
                "evidence": "Berhasil menangani...",
                "strong_signal": "Memahami arsitektur NestJS",
                "red_flag": "Belum menjelaskan testing edge-case"
            }},
            "problem_solving": {{
                "score": 6,
                "evidence": "Saat ditanya debugging...",
                "strong_signal": "None",
                "red_flag": "Terlalu bergantung pada senior"
            }},
            "culture_fit": {{
                "score": 8,
                "evidence": "Saat ada perbedaan pendapat...",
                "strong_signal": "Sikap sangat pragmatis",
                "red_flag": "None"
            }}
        }},
        "ai_evaluation_insight": {{
            "key_strengths": ["Strength 1", "Strength 2"],
            "growth_areas": ["Weakness 1", "Weakness 2"]
        }},
        "machine_recommendation": "RECOMMENDED FOR HIRE",
        "executive_summary": "1 kalimat ringkasan tegas."
    }}
    """

    try:
        resp = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict technical interview evaluator. You output valid JSON only."},
                {"role": "user", "content": prompt_analyzer}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        hasil = json.loads(resp.choices[0].message.content)
        save_result(payload, hasil)
        print(f"[BACKGROUND] Selesai analisa untuk Room: {payload.roomName}")

    except Exception as e:
        print(f"[BACKGROUND ERROR] Room: {payload.roomName} | Error: {e}")


# ==========================================
# 6. ENDPOINT API
# ==========================================
@app.post("/analyze-interview")
async def analyze_interview(payload: InterviewPayload, background_tasks: BackgroundTasks):
    if not payload.transcript:
        raise HTTPException(status_code=400, detail="Transcript kosong!")

    background_tasks.add_task(proses_dan_simpan, payload)

    return {
        "status": "success",
        "message": "Transcript diterima, analisa sedang berjalan di background.",
        "roomName": payload.roomName,
        "jobId": payload.jobId
    }
