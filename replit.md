# Resume ATS Optimizer

## Overview
An AI-powered Resume ATS Optimizer that analyzes job descriptions against resumes to identify skill gaps and automatically rewrites resumes to maximize ATS (Applicant Tracking System) match scores. Uses OpenAI via Replit AI Integrations (no API key needed).

## Project Architecture
- **Backend**: Python FastAPI application (`backend/main.py`)
  - Serves the frontend HTML from `docs/index.html`
  - API endpoints:
    - `GET /` - Serves frontend
    - `GET /health` - Health check
    - `POST /extract_pdf` - Extracts text from uploaded PDF resumes
    - `POST /analyze` - AI-powered skill gap analysis (identifies real skills, not filler words)
    - `POST /rewrite` - AI-powered line-by-line resume optimization
  - Uses pypdf for PDF text extraction
  - Uses OpenAI (gpt-5-mini) via Replit AI Integrations for all AI features
- **Frontend**: Single-page HTML app (`docs/index.html`)
  - Modern, clean UI with 3-step flow: Input -> Skill Gap Analysis -> AI Resume Optimization
  - Vanilla HTML/CSS/JS, no build step
  - Communicates with backend API using relative URLs (same-origin)

## Key Files
- `backend/main.py` - FastAPI server with all API routes and AI logic
- `docs/index.html` - Main frontend page (served by FastAPI)
- `frontend/index.html` - Legacy frontend (not used)
- `backend/requirements.txt` - Original requirements reference (deps managed via pyproject.toml)

## Running
- Single workflow: `python -m uvicorn backend.main:app --host 0.0.0.0 --port 5000 --reload`
- The backend serves both the API and the frontend on port 5000

## Environment Variables
- `AI_INTEGRATIONS_OPENAI_API_KEY` - Auto-set by Replit AI Integrations
- `AI_INTEGRATIONS_OPENAI_BASE_URL` - Auto-set by Replit AI Integrations

## Recent Changes
- 2026-02-17: Major upgrade to v2.0
  - Replaced rule-based keyword extraction with AI-powered skill identification (no more filler words)
  - Added line-by-line AI resume optimization with ATS scoring
  - Completely redesigned frontend with modern 3-step UI
  - Switched to Replit AI Integrations (gpt-5-mini model)
  - Added before/after ATS score comparison
  - Added copy and download functionality for optimized resumes
- 2026-02-17: Initial Replit setup
  - Configured FastAPI to serve frontend from docs/index.html
  - Updated CORS to allow all origins for Replit proxy
