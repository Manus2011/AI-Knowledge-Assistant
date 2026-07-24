from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.document_service import document_service
from app.services.parser_service import parse_document, ParsingError
from app.services.stats_service import get_document_stats
from app.services.categorization_service import categorization_service
from app.services.keyword_service import extract_keywords

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    doc = document_service.save_upload(
        filename=file.filename,
        content_type=file.content_type,
        contents=contents,
    )

    # try to parse it right away so we know if it's readable
    try:
        text = parse_document(doc.file_path)
        doc.extracted_text = text
        doc.category = categorization_service.categorize(text)

        # use other uploaded docs as context if there are enough of them,
        # otherwise fall back to the generic reference corpus
        other_docs = [
            d.extracted_text for d in document_service.list_documents()
            if d.id != doc.id and d.extracted_text
        ]
        comparison_corpus = other_docs if len(other_docs) >= 3 else None
        doc.keywords = extract_keywords(text, comparison_corpus=comparison_corpus)

        document_service.update_document(doc)
    except ParsingError as e:
        # still keep the upload, just flag that parsing failed
        return {
            "document": doc.to_dict(),
            "warning": f"File uploaded but couldn't be parsed: {str(e)}",
        }

    return {"document": doc.to_dict(), "preview": text[:300]}


@router.get("/")
def list_documents():
    docs = document_service.list_documents()
    return {"count": len(docs), "documents": [d.to_dict() for d in docs]}


@router.get("/stats/summary")
def document_stats():
    return get_document_stats()


@router.get("/{doc_id}")
def get_document(doc_id: str):
    doc = document_service.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc.to_dict()
