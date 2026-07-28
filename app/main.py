from fastapi import File, UploadFile, Form
from typing import List
import io
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

app = FastAPI(
    title="Resume Screener API",
    version="1.0",
    description="AI-powered resume screening using RAG"
)

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
except Exception as e:
    print(f"Error initializing screener: {e}")
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

    results = []

    for file in files:
        try:
            contents = await file.read()
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
    results.sort(key=lambda r: r["score"], reverse=True)

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)