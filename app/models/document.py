import uuid
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class Document:
    """Represents a single uploaded document and its metadata."""

    filename: str
    content_type: str
    size_bytes: int
    file_path: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    extracted_text: str | None = None
    category: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "uploaded_at": self.uploaded_at.isoformat(),
            "category": self.category,
            "has_extracted_text": self.extracted_text is not None,
        }
