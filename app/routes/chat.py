from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services import conversation_service as conv
from app.services.llm_service import llm_service

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None


@router.post("/")
def chat(req: ChatRequest):
    conversation_id = req.conversation_id or conv.new_conversation_id()
    history = conv.get_history(conversation_id)

    reply = llm_service.chat(req.message, history=history)

    conv.add_message(conversation_id, "user", req.message)
    conv.add_message(conversation_id, "assistant", reply)
    return {
        "conversation_id": conversation_id,
        "reply": reply,
        "provider": llm_service.provider.name,
    }


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str):
    return {"conversation_id": conversation_id, "messages": conv.get_history(conversation_id, limit=100)}
