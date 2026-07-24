import pandas as pd
from app.services.document_service import document_service


def get_document_stats() -> dict:
    # pandas practice from day 5, pulling basic stats off the registry
    docs = document_service.list_documents()

    if not docs:
        return {"total_documents": 0}

    df = pd.DataFrame([d.to_dict() for d in docs])
    df["uploaded_at"] = pd.to_datetime(df["uploaded_at"])

    stats = {
        "total_documents": len(df),
        "total_size_bytes": int(df["size_bytes"].sum()) if "size_bytes" in df else None,
        "avg_size_bytes": float(df["size_bytes"].mean()) if "size_bytes" in df else None,
        "by_content_type": df["content_type"].value_counts().to_dict(),
        "documents_with_category": int(df["category"].notna().sum()),
        "most_recent_upload": df["uploaded_at"].max().isoformat(),
    }
    return stats
