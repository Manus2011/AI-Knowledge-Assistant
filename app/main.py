import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core import config
from app.core.db import init_db
from app.routes import documents, chat, agent, auth

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")

DESCRIPTION = """
Upload documents, get them categorized and indexed, then ask questions about them.

* **auth**: register and log in to get a bearer token
* **documents**: upload (txt/pdf), categories, keywords, search, stats
* **chat**: plain chat and document chat (RAG) with sources and conversation memory
* **agent**: LangGraph agent that picks a tool; deleting asks for approval first

Without `ANTHROPIC_API_KEY` set, answers are extractive (sentences pulled from your documents).
"""

TAGS = [
    {"name": "auth", "description": "Register and log in"},
    {"name": "documents", "description": "Upload, search and manage documents"},
    {"name": "chat", "description": "Chat and document Q&A"},
    {"name": "agent", "description": "Tool-using agent with human approval"},
]

app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    description=DESCRIPTION,
    version="1.0.0",
    openapi_tags=TAGS,
)


@app.on_event("startup")
def startup():
    init_db()
    if config.JWT_SECRET == "dev-secret-change-me":
        log.warning("JWT_SECRET is the default, set a real one before deploying")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    log.info("%s %s %s -> %s %.1fms", request_id, request.method, request.url.path, response.status_code, ms)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    # log the real error, don't leak it to the client
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(agent.router)


@app.get("/", tags=["meta"])
def read_root():
    return {"status": "ok", "message": "AI Knowledge Assistant API running", "docs": "/docs"}


@app.get("/health", tags=["meta"])
def health_check():
    return {"status": "healthy"}
