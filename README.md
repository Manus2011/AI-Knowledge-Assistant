# Enterprise AI Knowledge Assistant

Capstone project for the Onshore Internship Program (8 weeks).
A production-style AI application supporting document upload, chat with documents,
semantic search, and RAG-based retrieval.

## Project Structure

```
app/
├── main.py          # FastAPI entrypoint
├── models/          # Pydantic + domain models
├── routes/          # API endpoint definitions
├── services/         # Business logic (parsing, upload, etc.)
└── utils/            # Shared helper functions
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

## Progress Log

- **Week 1, Day 1** — Repo structure, Git workflow, FastAPI skeleton
