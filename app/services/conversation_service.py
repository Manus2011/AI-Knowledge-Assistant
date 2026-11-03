import uuid
from datetime import datetime
from app.core.db import get_conn

MAX_HISTORY = 10  # last n messages sent to the model so the prompt doesn't grow forever


def new_conversation_id() -> str:
    return str(uuid.uuid4())


def add_message(conversation_id: str, role: str, content: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(conversation_id: str, limit: int = MAX_HISTORY) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
        (conversation_id, limit),
    ).fetchall()
    conn.close()
    history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    # the api wants the first message to be from the user
    while history and history[0]["role"] != "user":
        history.pop(0)
    return history
