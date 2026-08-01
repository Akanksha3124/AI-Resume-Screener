from fastapi import File, UploadFile, Form
from typing import List
import io
import logging
from app.database import init_db, SessionLocal, ScoringResult
from PyPDF2 import PdfReader
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.rag_engine import ResumeScreener
import os
from dotenv import load_dotenv

load_dotenv()

MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("resume_screener")

app = FastAPI(
    title="Resume Screener API",
    version="1.0",
    description="AI-powered resume screening using RAG"
)
init_db()

# Allow the frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/app")
def serve_frontend():
    return FileResponse("app/static/index.html")

# Initialize RAG engine
try:
    screener = ResumeScreener("data/resumes")
    logger.info("Screener initialized successfully")
except Exception as e:
    logger.error(f"Error initializing screener: {e}")
    screener = None

class ScoringRequest(BaseModel):
    job_description: str
    resume_text: str

class ScoringResponse(BaseModel):
    score: int
    reason: str
    match_keywords: list

@app.get("/")
def root():
    return {"message": "Resume Screener API v1.0", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "screener": "ready" if screener else "not_initialized"}

@app.post("/score", response_model=ScoringResponse)
def score_resume(request: ScoringRequest):
    """Score a resume against job description"""
    if not screener:
        raise HTTPException(status_code=503, detail="Screener not initialized")
    
    try:
        result = screener.score_candidate(
            request.job_description,
            request.resume_text
        )
        return ScoringResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score-resumes")
async def score_resumes(
    job_description: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload one or more PDF resumes and score them all against a job description."""
    if not screener:
        raise HTTPException(status_code=503, detail="Screener not initialized")

    # Validate job description
    if len(job_description.strip()) < 10:
        raise HTTPException(
            status_code=422,
            detail="Job description is too short. Please provide at least 10 characters."
        )

    if not files:
        raise HTTPException(status_code=422, detail="Please upload at least one file.")

    results = []
    logger.info(f"Scoring {len(files)} resume(s) against job description ({len(job_description)} chars)")

    for file in files:
        try:
            # Validate file type
            if not file.filename.lower().endswith(".pdf"):
                logger.warning(f"Rejected non-PDF file: {file.filename}")
                results.append({
                    "filename": file.filename,
                    "score": 0,
                    "reason": "File rejected: only PDF files are supported.",
                    "match_keywords": []
                })
                continue

            contents = await file.read()

            # Validate file size
            if len(contents) > MAX_FILE_SIZE_BYTES:
                logger.warning(f"Rejected oversized file: {file.filename} ({len(contents)} bytes)")
                results.append({
                    "filename": file.filename,
                    "score": 0,
                    "reason": f"File rejected: exceeds {MAX_FILE_SIZE_MB}MB size limit.",
                    "match_keywords": []
                })
                continue
            pdf_reader = PdfReader(io.BytesIO(contents))
            resume_text = ""
            for page in pdf_reader.pages:
                resume_text += page.extract_text() or ""

            if not resume_text.strip():
                results.append({
                    "filename": file.filename,
                    "score": 0,
                    "reason": "Could not extract text from this PDF",
                    "match_keywords": []
                })
                continue

            scored = screener.score_resume_direct(job_description, resume_text)
            results.append({
                "filename": file.filename,
                **scored
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "score": 0,
                "reason": f"Error processing file: {str(e)}",
                "match_keywords": []
            })

    # Rank by score, highest first
    # Rank by score, highest first
    results.sort(key=lambda r: r["score"], reverse=True)

    # Save each result to the database
    db = SessionLocal()
    try:
        for r in results:
            record = ScoringResult(
                filename=r["filename"],
                job_description=job_description,
                score=r["score"],
                reason=r["reason"],
                match_keywords=",".join(r.get("match_keywords", []))
            )
            db.add(record)
        db.commit()
        logger.info(f"Saved {len(results)} scoring result(s) to database")
    except Exception as e:
        logger.error(f"Failed to save results to database: {e}")
        db.rollback()
    finally:
        db.close()

    return {"results": results}

@app.get("/history")
def get_history(limit: int = 20):
    """View past scoring results, most recent first."""
    db = SessionLocal()
    try:
        records = db.query(ScoringResult).order_by(ScoringResult.created_at.desc()).limit(limit).all()
        return {
            "results": [
                {
                    "id": r.id,
                    "filename": r.filename,
                    "score": r.score,
                    "reason": r.reason,
                    "match_keywords": r.match_keywords.split(",") if r.match_keywords else [],
                    "created_at": r.created_at.isoformat()
                }
                for r in records
            ]
        }
    finally:
        db.close()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)