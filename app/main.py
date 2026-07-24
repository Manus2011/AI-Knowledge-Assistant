from fastapi import FastAPI

from app.routes import documents

app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    description="Capstone project - internship program",
    version="0.1.0",
)

app.include_router(documents.router)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Knowledge Assistant API running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
