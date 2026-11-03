from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.deps import current_user
from app.services.agent_service import agent_service

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool


@router.post("/run")
def run_agent(req: AgentRequest, user: str = Depends(current_user)):
    return agent_service.run(req.message, user, req.thread_id)


@router.post("/approve")
def approve(req: ApprovalRequest, user: str = Depends(current_user)):
    try:
        return agent_service.resume(req.thread_id, req.approved, user)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
