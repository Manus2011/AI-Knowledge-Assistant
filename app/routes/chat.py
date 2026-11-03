from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.deps import current_user
from app.services import conversation_service as conv
from app.services import rag_service
from app.services.llm_service import llm_service

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None


def _key(user: str, conversation_id: str) -> str:
    # conversations are stored per user so ids can't be guessed across accounts
    return f"{user}:{conversation_id}"


@router.post("/")
def chat(req: ChatRequest, user: str = Depends(current_user)):
    conversation_id = req.conversation_id or conv.new_conversation_id()
    key = _key(user, conversation_id)
    history = conv.get_history(key)

    reply = llm_service.chat(req.message, history=history)

    conv.add_message(key, "user", req.message)
    conv.add_message(key, "assistant", reply)
    return {"conversation_id": conversation_id, "reply": reply,
            "provider": llm_service.provider.name}


@router.post("/documents")
def chat_with_documents(req: ChatRequest, user: str = Depends(current_user)):
    # same as /chat but answers are grounded in retrieved chunks, with sources
    conversation_id = req.conversation_id or conv.new_conversation_id()
    key = _key(user, conversation_id)
    history = conv.get_history(key)
    result = rag_service.answer(req.message, history=history, owner=user)
    conv.add_message(key, "user", req.message)
    conv.add_message(key, "assistant", result["answer"])
    return {"conversation_id": conversation_id, "reply": result["answer"],
            "sources": result["sources"], "provider": llm_service.provider.name}


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, user: str = Depends(current_user)):
    return {"conversation_id": conversation_id,
            "messages": conv.get_history(_key(user, conversation_id), limit=100)}
