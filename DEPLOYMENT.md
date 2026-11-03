# Deployment

## Local with Docker

```bash
cp .env.example .env     # set JWT_SECRET (long random string) and optionally ANTHROPIC_API_KEY
docker compose up --build
```

API at `http://localhost:8000`, docs at `/docs`. Uploaded files and the sqlite db live in the
`app_data` volume so they survive restarts. Note: I wrote these files but couldn't run docker
where I built the project, so if the build fails check the python version / requirements first.

## Without Docker

```bash
pip install -r requirements.txt
JWT_SECRET=some-long-random-string uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Config (env vars)

| var | default | what |
|---|---|---|
| JWT_SECRET | dev-secret-change-me | signs login tokens, must change in prod |
| JWT_EXPIRE_MINUTES | 60 | token lifetime |
| ANTHROPIC_API_KEY | empty | enables real LLM answers, otherwise extractive |
| LLM_MODEL | claude-sonnet-4-5 | model name |
| CATEGORIZER | logreg | logreg or mlp |
| DATA_DIR / DB_PATH | ./data | where files and sqlite db go |
| CHUNK_SIZE / CHUNK_OVERLAP / TOP_K | 500 / 80 / 4 | retrieval settings |

## Before putting it on a real server

- put it behind https (nginx / a cloud load balancer), tokens are bearer tokens
- run a single worker, or move agent state off the in-memory checkpointer first
- sqlite is fine for a demo, switch to postgres for multiple instances
- back up the data volume
