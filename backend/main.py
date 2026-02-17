import os
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import io
from fastapi import Form, FastAPI, UploadFile, File, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from openai import OpenAI

app = FastAPI(title="Resume Keyword Matcher API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
# Using Replit AI Integrations for OpenAI access
AI_INTEGRATIONS_OPENAI_API_KEY = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
AI_INTEGRATIONS_OPENAI_BASE_URL = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")

client = OpenAI(
    api_key=AI_INTEGRATIONS_OPENAI_API_KEY,
    base_url=AI_INTEGRATIONS_OPENAI_BASE_URL,
)

MODEL = "gpt-5-mini"

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "docs"


def extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages).strip()


def parse_json_from_text(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    first = text.find("[") if text.find("[") != -1 and (text.find("{") == -1 or text.find("[") < text.find("{")) else text.find("{")
    last_bracket = text.rfind("]")
    last_brace = text.rfind("}")
    last = max(last_bracket, last_brace)
    if first != -1 and last != -1:
        text = text[first:last + 1]
    return json.loads(text)


class AnalyzeRequest(BaseModel):
    jd: str
    resume: str

class AnalyzeResponse(BaseModel):
    score_pct: int
    found: List[str]
    missing: List[str]
    all_skills: List[str]

class RewriteRequest(BaseModel):
    jd: str
    resume: str
    missing_skills: List[str] = Field(default_factory=list)

class RewriteResponse(BaseModel):
    optimized_resume: str
    changes_made: List[Dict[str, str]]
    ats_score_before: int
    ats_score_after: int
    tips: List[str]


@app.get("/")
def root():
    index_file = FRONTEND_DIR / "index.html"
    return FileResponse(str(index_file), media_type="text/html", headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health():
    return {"status": "ok", "service": "resume-matcher", "version": "2.0"}


@app.post("/extract_pdf")
async def extract_pdf(resume_pdf: UploadFile = File(...)):
    data = await resume_pdf.read()
    text = extract_pdf_text(data)
    if not text.strip():
        return {"error": "Could not extract text from this PDF. It may be a scanned image. Try a text-based PDF."}
    return {"text": text}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not AI_INTEGRATIONS_OPENAI_API_KEY or not AI_INTEGRATIONS_OPENAI_BASE_URL:
        raise HTTPException(status_code=503, detail="AI service is not configured. Please set up OpenAI integration.")

    if len(req.jd) > 50000 or len(req.resume) > 50000:
        raise HTTPException(status_code=400, detail="Input too large. Please paste less text.")

    prompt = f"""Analyze this job description and resume. Extract ONLY real, specific skills, technologies, tools, certifications, methodologies, and technical competencies from the job description. 

DO NOT include generic words like: experience, team, communication, responsibilities, requirements, ability, knowledge, understanding, working, management (unless it's a specific methodology like "project management" or "change management"), about, strong, excellent, preferred, required, etc.

ONLY include items like: Python, JavaScript, AWS, Agile, Scrum, SQL, Docker, Kubernetes, React, machine learning, data analysis, CI/CD, REST APIs, project management, Six Sigma, Tableau, Excel, etc.

JOB DESCRIPTION:
{req.jd}

RESUME:
{req.resume}

Return STRICT JSON (no markdown, no explanation) with this exact structure:
{{
  "all_skills": ["skill1", "skill2", ...],
  "found": ["skill1", "skill2", ...],
  "missing": ["skill1", "skill2", ...],
  "score_pct": 75
}}

Where:
- all_skills: every real skill/technology/tool/certification found in the job description (max 40)
- found: skills from the JD that ARE present in the resume
- missing: skills from the JD that are NOT in the resume
- score_pct: percentage of JD skills found in resume (0-100)"""

    try:
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert ATS (Applicant Tracking System) analyzer. You identify only real, actionable skills, technologies, tools, and certifications. Never include generic filler words. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=8192,
        )

        data = json.loads(resp.choices[0].message.content)
        return AnalyzeResponse(
            score_pct=data.get("score_pct", 0),
            found=data.get("found", []),
            missing=data.get("missing", []),
            all_skills=data.get("all_skills", []),
        )

    except Exception as e:
        print(f"ANALYZE ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")


@app.post("/rewrite")
def rewrite_resume(req: RewriteRequest) -> Dict[str, Any]:
    if not AI_INTEGRATIONS_OPENAI_API_KEY or not AI_INTEGRATIONS_OPENAI_BASE_URL:
        return {"error": "AI service is not configured. Please set up OpenAI integration."}

    if len(req.jd) > 50000 or len(req.resume) > 50000:
        return {"error": "Input too large. Please paste less text."}

    missing_str = ", ".join(req.missing_skills[:50]) if req.missing_skills else "none identified"

    prompt = f"""You are an expert ATS resume optimizer. Your job is to rewrite a candidate's resume to maximize their ATS (Applicant Tracking System) match score for a specific job.

JOB DESCRIPTION:
{req.jd}

ORIGINAL RESUME:
{req.resume}

MISSING SKILLS/KEYWORDS:
{missing_str}

Instructions:
1. Go through the resume LINE BY LINE
2. Rewrite bullet points to naturally incorporate missing keywords where truthful
3. Strengthen action verbs and quantify achievements where possible
4. Reorganize sections to prioritize the most relevant experience
5. Add a tailored professional summary if one doesn't exist
6. DO NOT fabricate experience or skills the candidate doesn't have — instead, reframe existing experience to better highlight relevant transferable skills
7. Keep the same general structure (sections, job titles, dates) but improve the content

Return STRICT JSON with this structure:
{{
  "optimized_resume": "The full rewritten resume as plain text with proper formatting",
  "changes_made": [
    {{"section": "Professional Summary", "original": "old text or N/A", "improved": "new text", "reason": "why this change helps"}},
    {{"section": "Experience - Job Title", "original": "old bullet", "improved": "new bullet", "reason": "why"}}
  ],
  "ats_score_before": 45,
  "ats_score_after": 82,
  "tips": [
    "Additional tip 1 for the candidate",
    "Additional tip 2"
  ]
}}

Make changes_made contain the 8-12 most impactful changes. Keep tips to 3-5 actionable items."""

    try:
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a professional ATS resume optimization expert. You rewrite resumes to maximize match scores while keeping content truthful. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=8192,
        )

        data = json.loads(resp.choices[0].message.content)
        return {"ok": True, "data": data}

    except Exception as e:
        print(f"REWRITE ERROR: {e}")
        return {"error": "Resume optimization failed. Please try again.", "details": str(e)}
