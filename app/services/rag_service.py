from app.core import config
from app.services import vector_store
from app.services.llm_service import llm_service

NOT_FOUND = "I couldn't find anything in your documents that answers that."


def answer(question: str, history: list[dict] | None = None, owner: str | None = None,
           top_k: int | None = None) -> dict:
    hits = vector_store.search(question, top_k=top_k or config.TOP_K, owner=owner)
    if not hits:
        # don't bother the model when retrieval found nothing
        return {"answer": NOT_FOUND, "sources": []}

    context = "\n\n".join(f"[{i + 1}] ({h['filename']}) {h['text']}" for i, h in enumerate(hits))
    reply = llm_service.chat(question, history=history, context=context)
    sources = [
        {"ref": i + 1, "doc_id": h["doc_id"], "filename": h["filename"],
         "chunk_index": h["chunk_index"], "score": h["score"]}
        for i, h in enumerate(hits)
    ]
    return {"answer": reply, "sources": sources}
