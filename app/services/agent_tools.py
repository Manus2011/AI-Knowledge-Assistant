# plain python functions the agent can call. kept separate from the graph so they are easy to test
from app.services.document_service import document_service
from app.services import vector_store
from app.services.stats_service import get_document_stats


def search_documents(query: str) -> dict:
    hits = vector_store.search(query, top_k=3)
    return {"hits": hits}


def list_documents() -> dict:
    docs = document_service.list_documents()
    return {"documents": [{"id": d.id, "filename": d.filename, "category": d.category} for d in docs]}


def document_stats() -> dict:
    return get_document_stats()


def find_document(name_hint: str):
    # match a filename mentioned in the user's message
    hint = name_hint.lower()
    for d in document_service.list_documents():
        if d.filename.lower() in hint or d.filename.lower().rsplit(".", 1)[0] in hint:
            return d
    return None


def delete_document(doc_id: str) -> dict:
    return {"deleted": document_service.delete_document(doc_id), "doc_id": doc_id}
