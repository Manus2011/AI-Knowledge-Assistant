from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.agent_service import agent_service

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool


@router.post("/run")
def run_agent(req: AgentRequest):
    return agent_service.run(req.message, req.thread_id)


@router.post("/approve")
def approve(req: ApprovalRequest):
    try:
        return agent_service.resume(req.thread_id, req.approved)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
