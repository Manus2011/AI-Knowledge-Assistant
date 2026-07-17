from fastapi import FastAPI

app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    description="Capstone project - internship program",
    version="0.1.0",
)


@app.get("/")
def read_root():
    """Health check endpoint - confirms the API is alive."""
    return {"status": "ok", "message": "AI Knowledge Assistant API running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
