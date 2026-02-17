# Resume Keyword Matcher

## Overview
A Resume Keyword Matcher tool that analyzes job descriptions against resumes to identify matching and missing keywords. It supports both text paste and PDF upload for resumes, and includes AI-powered suggestions via OpenAI.

## Project Architecture
- **Backend**: Python FastAPI application (`backend/main.py`)
  - Serves the frontend HTML from `docs/index.html`
  - Provides REST API endpoints: `/analyze`, `/analyze_pdf`, `/suggest`
  - Uses pypdf for PDF text extraction
  - Uses OpenAI API for AI-powered resume suggestions (optional, requires OPENAI_API_KEY)
- **Frontend**: Single-page HTML app (`docs/index.html`)
  - Vanilla HTML/CSS/JS, no build step
  - Communicates with backend API using relative URLs

## Key Files
- `backend/main.py` - FastAPI server with all API routes and static file serving
- `backend/requirements.txt` - Original requirements reference (deps managed via pyproject.toml)
- `docs/index.html` - Main frontend page (served by FastAPI)
- `frontend/index.html` - Simpler version of frontend (not used in production)

## Running
- Single workflow: `python -m uvicorn backend.main:app --host 0.0.0.0 --port 5000 --reload`
- The backend serves both the API and the frontend on port 5000

## Environment Variables
- `OPENAI_API_KEY` (optional) - Required for AI suggestion feature (`/suggest` endpoint)

## Recent Changes
- 2026-02-17: Initial Replit setup
  - Configured FastAPI to serve frontend from docs/index.html
  - Updated CORS to allow all origins for Replit proxy
  - Changed API_BASE to relative URLs (same-origin)
  - Removed Render-specific references
