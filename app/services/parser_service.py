import os
from pypdf import PdfReader


class ParsingError(Exception):
    """Raised when a document can't be parsed into text."""
    pass


def parse_document(file_path: str) -> str:
    """Extract raw text from a document. Supports .txt and .pdf for now.

    Word docs and other formats can get added later, most of what we
    need for the RAG pipeline is just readable text out of this.
    """
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif ext == ".pdf":
            reader = PdfReader(file_path)
            text_chunks = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_chunks.append(page_text)
            return "\n".join(text_chunks)

        else:
            raise ParsingError(f"Unsupported file type: {ext}")

    except FileNotFoundError:
        raise ParsingError(f"File not found: {file_path}")
    except UnicodeDecodeError:
        raise ParsingError(f"Could not decode {file_path}, file may be corrupted or wrong encoding")
    except Exception as e:
        # catch-all so a single bad file can't crash the whole upload flow
        raise ParsingError(f"Failed to parse {file_path}: {str(e)}")
