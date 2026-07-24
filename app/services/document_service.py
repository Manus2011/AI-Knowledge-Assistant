import os
import json
from app.models.document import Document

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
REGISTRY_PATH = os.path.join(DATA_DIR, "registry.json")


class DocumentService:
    # using a json file as a registry for now instead of a real db,
    # good enough until we get to the db/auth stuff later on

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self._documents: dict[str, Document] = {}
        self._load_registry()

    def _load_registry(self):
        if not os.path.exists(REGISTRY_PATH):
            return
        with open(REGISTRY_PATH, "r") as f:
            raw = json.load(f)
        for item in raw:
            doc = Document(
                filename=item["filename"],
                content_type=item["content_type"],
                size_bytes=item["size_bytes"],
                file_path=item["file_path"],
                id=item["id"],
            )
            doc.category = item.get("category")
            self._documents[doc.id] = doc

    def _save_registry(self):
        with open(REGISTRY_PATH, "w") as f:
            json.dump([d.to_dict() | {"file_path": d.file_path} for d in self._documents.values()], f, indent=2)

    def save_upload(self, filename: str, content_type: str, contents: bytes) -> Document:
        file_path = os.path.join(DATA_DIR, filename)

        # avoid overwriting a file with the same name
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(DATA_DIR, f"{base}_{counter}{ext}")
            counter += 1

        with open(file_path, "wb") as f:
            f.write(contents)

        doc = Document(
            filename=os.path.basename(file_path),
            content_type=content_type,
            size_bytes=len(contents),
            file_path=file_path,
        )
        self._documents[doc.id] = doc
        self._save_registry()
        return doc

    def update_document(self, doc: Document):
        # saves changes made after the initial upload (category, keywords, etc)
        self._documents[doc.id] = doc
        self._save_registry()

    def list_documents(self) -> list[Document]:
        return list(self._documents.values())

    def get_document(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)


document_service = DocumentService()
