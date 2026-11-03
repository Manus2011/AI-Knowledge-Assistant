from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from app.core.deps import current_user
from app.services.document_service import document_service
from app.services.parser_service import parse_document, ParsingError
from app.services.stats_service import get_document_stats
from app.services.categorization_service import categorization_service
from app.services.keyword_service import extract_keywords
from app.services import vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


def _owned(doc_id: str, user: str):
    # other people's documents look like they don't exist
    doc = document_service.get_document(doc_id)
    if doc is None or doc.owner != user:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), user: str = Depends(current_user)):
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    doc = document_service.save_upload(
        filename=file.filename,
        content_type=file.content_type,
        contents=contents,
        owner=user,
    )

    # parse right away so we know if the file is even readable
    try:
        text = parse_document(doc.file_path)
        doc.extracted_text = text
        doc.category = categorization_service.categorize(text)

        # use the user's other docs for comparison once there's enough of them
        other_docs = [
            d.extracted_text for d in document_service.list_documents(owner=user)
            if d.id != doc.id and d.extracted_text
        ]
        comparison_corpus = other_docs if len(other_docs) >= 3 else None
        doc.keywords = extract_keywords(text, comparison_corpus=comparison_corpus)

        document_service.update_document(doc)
        n_chunks = vector_store.index_document(doc.id, text)
    except ParsingError as e:
        # keep the upload but flag the parse failure
        return {
            "document": doc.to_dict(),
            "warning": f"File uploaded but couldn't be parsed: {str(e)}",
        }

    return {"document": doc.to_dict(), "preview": text[:300], "chunks_indexed": n_chunks}


@router.get("/")
def list_documents(user: str = Depends(current_user)):
    docs = document_service.list_documents(owner=user)
    return {"count": len(docs), "documents": [d.to_dict() for d in docs]}


@router.get("/search")
def search_documents(q: str, top_k: int = 4, user: str = Depends(current_user)):
    return {"query": q, "results": vector_store.search(q, top_k=top_k, owner=user)}


@router.get("/stats/summary")
def document_stats(user: str = Depends(current_user)):
    return get_document_stats(owner=user)


@router.get("/{doc_id}")
def get_document(doc_id: str, user: str = Depends(current_user)):
    return _owned(doc_id, user).to_dict()


@router.delete("/{doc_id}")
def delete_document(doc_id: str, user: str = Depends(current_user)):
    _owned(doc_id, user)
    document_service.delete_document(doc_id)
    return {"deleted": doc_id}
