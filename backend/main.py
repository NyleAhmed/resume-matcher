import os
import re
import json
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader

from openai import OpenAI

# -------------------------
# App + CORS
# -------------------------
app = FastAPI(title="Resume Keyword Matcher API", version="1.0.0")

# For a quick demo we allow all origins.
# Later you can restrict to your GitHub Pages domain:
#   ["https://nyleahmed.github.io"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# OpenAI Client
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# -------------------------
# Helpers (keyword extraction)
# -------------------------
STOPWORDS = set([
    "a","an","the","and","or","but","if","then","else","when","while","to","of","in","on","for","from",
    "with","without","at","by","as","is","are","was","were","be","been","being","it","its","this","that",
    "these","those","you","your","we","our","they","their","i","me","my","us","them",
    "will","can","may","might","should","must","do","does","did","doing","done",
    "not","no","yes","so","than","too","very","more","most","less","least",
    "role","job","work","team","teams","responsibilities","responsibility","experience","required","requirements",
    "preferred","skills","skill","ability","abilities","including","include","plus"
])

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("\u2019", "'")
    s = re.sub(r"[^a-z0-9\s'+-]", " ", s)   # keep letters, digits, space, +, -, '
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_words(text: str) -> List[str]:
    return [w for w in normalize_text(text).split(" ") if w]

def build_word_freq(words: List[str], min_len: int) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for w in words:
        if len(w) < min_len:
            continue
        if w in STOPWORDS:
            continue
        token = re.sub(r"^'+|'+$", "", w)
        if not token or len(token) < min_len:
            continue
        freq[token] = freq.get(token, 0) + 1
    return freq

def build_bigram_freq(words: List[str], min_len: int) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for i in range(len(words) - 1):
        a, b = words[i], words[i + 1]
        if len(a) < min_len or len(b) < min_len:
            continue
        if a in STOPWORDS or b in STOPWORDS:
            continue
        phrase = f"{a} {b}"
        freq[phrase] = freq.get(phrase, 0) + 1
    return freq

def top_keywords(freq: Dict[str, int], top_n: int) -> List[str]:
    items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in items[:top_n]]

def term_in_text(term: str, resume_text_normalized: str) -> bool:
    if " " in term:
        return term in resume_text_normalized
    return f" {resume_text_normalized} ".find(f" {term} ") != -1


# -------------------------
# Schemas
# -------------------------
class AnalyzeRequest(BaseModel):
    jd: str
    resume: str
    minLen: int = 3
    topN: int = 40
    includeBigrams: bool = True

class AnalyzeResponse(BaseModel):
    scorePct: int
    found: List[str]
    missing: List[str]
    keywords: List[str]

class SuggestRequest(BaseModel):
    jd: str = Field(..., description="Job description text")
    resume: str = Field(..., description="Resume text")
    missing: List[str] = Field(default_factory=list, description="Missing keywords (optional, from matcher)")
    target_role: Optional[str] = Field(default=None, description="Optional role name to tailor suggestions")
    level: str = Field(default="intern", description="intern / entry / mid / senior")
    tone: str = Field(default="confident", description="confident / formal / concise")
    max_bullets: int = Field(default=6, ge=3, le=12)

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "resume-matcher"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    jd_words = tokenize_words(req.jd)
    resume_norm = normalize_text(req.resume)

    word_freq = build_word_freq(jd_words, req.minLen)
    if req.includeBigrams:
        bigram_freq = build_bigram_freq(jd_words, req.minLen)
        # merge
        for k, v in bigram_freq.items():
            word_freq[k] = word_freq.get(k, 0) + v

    keywords = top_keywords(word_freq, req.topN)

    found, missing = [], []
    for term in keywords:
        if term_in_text(term, resume_norm):
            found.append(term)
        else:
            missing.append(term)

    total = max(1, len(keywords))
    score_pct = round((len(found) / total) * 100)

    return AnalyzeResponse(
        scorePct=score_pct,
        found=found,
        missing=missing,
        keywords=keywords
    )

@app.post("/analyze_pdf", response_model=AnalyzeResponse)
async def analyze_pdf(
    jd: str,
    minLen: int = 3,
    topN: int = 40,
    includeBigrams: bool = True,
    resume_pdf: UploadFile = File(...)
):
    # Extract text from PDF
    data = await resume_pdf.read()
    reader = PdfReader(io_bytes := __import__("io").BytesIO(data))
    pages_text = []
    for p in reader.pages:
        pages_text.append(p.extract_text() or "")
    resume_text = "\n".join(pages_text)

    req = AnalyzeRequest(jd=jd, resume=resume_text, minLen=minLen, topN=topN, includeBigrams=includeBigrams)
    return analyze(req)

@app.post("/suggest")
def suggest(req: SuggestRequest) -> Dict[str, Any]:
    if not client:
        return {
            "error": "OPENAI_API_KEY is not set on the server.",
            "how_to_fix": "Set OPENAI_API_KEY in your Render Environment Variables and redeploy."
        }

    # Basic size guard (avoid accidentally sending huge PDFs)
    if len(req.jd) > 50000 or len(req.resume) > 50000:
        return {"error": "Input too large. Please paste less text / use a shorter resume extract."}

    missing_list = req.missing[:80]  # keep it sane

    system = (
        "You are an expert ATS-friendly resume coach and recruiter. "
        "You give specific, actionable improvements without lying or inventing experience. "
        "If the user lacks a skill, suggest phrasing that shows learning/projects instead of claiming it."
    )

    # Ask for STRICT JSON so frontend can render nicely
    user = {
        "task": "Review resume vs job description. Return ATS + recruiter improvements.",
        "target_role": req.target_role or "not provided",
        "level": req.level,
        "tone": req.tone,
        "max_bullets": req.max_bullets,
        "job_description": req.jd,
        "resume": req.resume,
        "missing_keywords_from_matcher": missing_list,
        "output_format": {
            "type": "json",
            "required_keys": [
                "headline",
                "overall_feedback",
                "top_gaps",
                "keyword_plan",
                "rewrite_examples",
                "ats_checks",
                "next_steps"
            ],
            "rewrite_examples_format": {
                "instruction": "Provide bullet rewrites as BEFORE/AFTER pairs.",
                "fields": ["before", "after", "why_it_works"]
            }
        }
    }

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)}
        ],
        temperature=0.3,
    )

    text = (resp.output_text or "").strip()

    # Parse JSON safely (model should return JSON, but we guard anyway)
    try:
        # If it returned extra text, attempt to extract JSON block
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            text = text[first:last+1]
        data = json.loads(text)
        return {"ok": True, "data": data}
    except Exception:
        return {"ok": True, "data": {"raw": (resp.output_text or "").strip()} }
