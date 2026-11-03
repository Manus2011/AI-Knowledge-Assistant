import os
import json
from datetime import datetime
from app.core import config
from app.core.db import get_conn, init_db
from app.models.document import Document


class DocumentService:
    # documents live in sqlite now, the old json registry lost keywords/text on restart

    def __init__(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        init_db()

    def _row_to_doc(self, row) -> Document:
        doc = Document(
            filename=row["filename"],
            content_type=row["content_type"],
            size_bytes=row["size_bytes"],
            file_path=row["file_path"],
            id=row["id"],
        )
        doc.uploaded_at = datetime.fromisoformat(row["uploaded_at"])
        doc.extracted_text = row["extracted_text"]
        doc.category = row["category"]
        doc.keywords = json.loads(row["keywords"]) if row["keywords"] else []
        doc.owner = row["owner"]
        return doc

    def save_upload(self, filename: str, content_type: str, contents: bytes, owner: str | None = None) -> Document:
        # strip any path bits from the client supplied name
        filename = os.path.basename(filename or "upload")
        file_path = os.path.join(config.DATA_DIR, filename)

        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(config.DATA_DIR, f"{base}_{counter}{ext}")
            counter += 1

        with open(file_path, "wb") as f:
            f.write(contents)

        doc = Document(
            filename=os.path.basename(file_path),
            content_type=content_type,
            size_bytes=len(contents),
            file_path=file_path,
        )
        doc.owner = owner
        conn = get_conn()
        conn.execute(
            "INSERT INTO documents (id, filename, content_type, size_bytes, file_path, uploaded_at, owner) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc.id, doc.filename, doc.content_type, doc.size_bytes, doc.file_path,
             doc.uploaded_at.isoformat(), owner),
        )
        conn.commit()
        conn.close()
        return doc

    def update_document(self, doc: Document):
        conn = get_conn()
        conn.execute(
            "UPDATE documents SET extracted_text = ?, category = ?, keywords = ? WHERE id = ?",
            (doc.extracted_text, doc.category, json.dumps(doc.keywords), doc.id),
        )
        conn.commit()
        conn.close()

    def list_documents(self, owner: str | None = None) -> list[Document]:
        conn = get_conn()
        if owner is None:
            rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM documents WHERE owner = ? ORDER BY uploaded_at", (owner,)
            ).fetchall()
        conn.close()
        return [self._row_to_doc(r) for r in rows]

    def get_document(self, doc_id: str) -> Document | None:
        conn = get_conn()
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        conn.close()
        return self._row_to_doc(row) if row else None

    def delete_document(self, doc_id: str) -> bool:
        doc = self.get_document(doc_id)
        if doc is None:
            return False
        conn = get_conn()
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()
        if doc.file_path and os.path.exists(doc.file_path):
            os.remove(doc.file_path)
        return True


document_service = DocumentService()
