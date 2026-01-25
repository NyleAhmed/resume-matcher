from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re
import io

import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(title="Resume Keyword Matcher API")

# Allow local frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    s = re.sub(r"[^a-z0-9\s'+-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # normalize common variants
    s = re.sub(r"\bpower\s*bi\b", "powerbi", s)
    s = re.sub(r"\bpost\s*gre\s*sql\b", "postgresql", s)
    return s

def token_pattern(min_len: int) -> str:
    return rf"(?u)\b[a-z0-9][a-z0-9'+-]{{{min_len-1},}}\b"

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
    total: int

def analyze_text(jd: str, resume: str, min_len: int, top_n: int, include_bigrams: bool) -> AnalyzeResponse:
    jd_norm = normalize_text(jd)
    resume_norm = normalize_text(resume)

    min_len = max(2, min(20, int(min_len)))
    top_n = max(10, min(200, int(top_n)))

    ngram_range = (1, 2) if include_bigrams else (1, 1)

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        token_pattern=token_pattern(min_len),
        lowercase=False
    )

    X = vectorizer.fit_transform([jd_norm])
    terms = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]

    filtered = []
    for term, score in zip(terms, scores):
        if score <= 0:
            continue
        if " " in term:
            a, b = term.split(" ", 1)
            if a in STOPWORDS or b in STOPWORDS:
                continue
        else:
            if term in STOPWORDS:
                continue
        filtered.append((term, float(score)))

    filtered.sort(key=lambda x: x[1], reverse=True)
    keywords = [t for (t, _) in filtered[:top_n]]

    padded_resume = f" {resume_norm} "
    found, missing = [], []
    for term in keywords:
        needle = f" {term} "
        if needle in padded_resume:
            found.append(term)
        else:
            missing.append(term)

    total = len(keywords) if keywords else 1
    score_pct = round((len(found) / total) * 100)

    return AnalyzeResponse(scorePct=score_pct, found=found, missing=missing, total=len(keywords))

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    return "\n".join(text_parts).strip()

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    return analyze_text(req.jd, req.resume, req.minLen, req.topN, req.includeBigrams)

@app.post("/analyze_pdf", response_model=AnalyzeResponse)
async def analyze_pdf(
    jd: str = Form(...),
    minLen: int = Form(3),
    topN: int = Form(40),
    includeBigrams: bool = Form(True),
    resume_pdf: UploadFile = File(...),
):
    pdf_bytes = await resume_pdf.read()
    resume_text = extract_text_from_pdf(pdf_bytes)

    if not resume_text:
        # Some PDFs are image-scans and have no selectable text.
        # This tool won't work for those without OCR.
        return AnalyzeResponse(scorePct=0, found=[], missing=[], total=0)

    return analyze_text(jd, resume_text, minLen, topN, includeBigrams)