from fastapi import APIRouter, Depends

from app.core.deps import current_user
from app.routes.chat import ChatRequest, _key
from app.services import chain_service
from app.services import conversation_service as conv

router = APIRouter(prefix="/chain", tags=["chat"])


@router.post("/ask")
def ask(req: ChatRequest, user: str = Depends(current_user)):
    # same idea as /chat/documents but built as a langchain chain with message history
    conversation_id = req.conversation_id or conv.new_conversation_id()
    result = chain_service.ask(req.message, _key(user, conversation_id), owner=user)
    return {"conversation_id": conversation_id, "reply": result["answer"], "sources": result["sources"]}
