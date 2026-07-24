# Enterprise AI Knowledge Assistant

Capstone project for the Onshore Internship Program (8 weeks).
A production-style AI application supporting document upload, chat with documents,
semantic search, and RAG-based retrieval.

## Project Structure

```
app/
├── main.py          # FastAPI entrypoint
├── models/          # Document model (OOP)
├── routes/          # API endpoint definitions
├── services/        # Business logic: upload, parsing, stats, categorization, keywords
└── utils/           # Shared helper functions
exercises/           # Standalone daily hands-on scripts from the weekly curriculum
tests/                # Unit and integration tests
data/                 # Uploaded documents (gitignored)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs.

## Features so far

- Upload documents (.txt, .pdf) via REST API
- Automatic text extraction/parsing
- Auto-categorization (resume / meeting_notes / report / contract / other)
- Auto keyword extraction using TF-IDF
- Document stats summary (pandas-based)

## Progress Log

- **Week 1** — Repo structure, Git workflow, FastAPI skeleton, document upload + parsing (txt/pdf), pandas-based stats endpoint
- **Week 2** — ML exercises (spam classifier, customer prediction, data viz), document categorization wired into the app
- **Week 3** — NLP exercises (resume classifier, sentiment analyzer, keyword extractor), automatic keyword extraction wired into document upload

