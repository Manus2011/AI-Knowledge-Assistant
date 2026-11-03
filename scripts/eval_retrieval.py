# quick retrieval check on a small hand written set. temp db so it doesn't touch real data
import os, sys, tempfile
tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = tmp
os.environ["DB_PATH"] = os.path.join(tmp, "eval.db")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import init_db, get_conn
from app.services import vector_store

DOCS = {
    "vacation": "Employees receive 20 days of paid vacation each year. Requests must be approved by a manager two weeks in advance. Unused days carry over up to five days. Sick leave is tracked separately and does not reduce vacation balance.",
    "expenses": "Travel expenses must be submitted within 30 days with receipts. Meals are reimbursed up to 50 dollars per day. Flights should be booked in economy class. Hotel stays above 200 dollars per night need director approval.",
    "security": "All laptops must use full disk encryption. Passwords must be rotated every 90 days. Multi factor authentication is required for the VPN. Lost devices must be reported to IT within 24 hours.",
    "onboarding": "New hires complete orientation during their first week. Each hire is paired with a mentor for 90 days. Equipment is shipped to remote employees before the start date. Benefits enrollment closes 30 days after hire.",
    "remote_work": "Employees may work remotely up to three days per week. Core hours are 10am to 3pm in the local time zone. Home office stipends are 500 dollars one time. Remote staff must be reachable on chat during core hours.",
}
QUERIES = [
    ("how many vacation days do I get", "vacation"), ("can I carry over unused vacation", "vacation"),
    ("who approves my time off", "vacation"), ("how long to submit travel receipts", "expenses"),
    ("what is the daily meal limit", "expenses"), ("do hotels need approval", "expenses"),
    ("how often must passwords change", "security"), ("do I need MFA for the VPN", "security"),
    ("what if my laptop is lost", "security"), ("how long is the mentor program", "onboarding"),
    ("when does benefits enrollment close", "onboarding"), ("when is equipment shipped to new hires", "onboarding"),
    ("how many days can I work from home", "remote_work"), ("what are the core hours", "remote_work"),
    ("is there a home office stipend", "remote_work"),
]

init_db()
conn = get_conn()
for name, text in DOCS.items():
    conn.execute("INSERT INTO documents (id, filename, uploaded_at) VALUES (?, ?, 'x')", (name, name + ".txt"))
conn.commit(); conn.close()
for name, text in DOCS.items():
    vector_store.index_document(name, text)

top1 = top3 = 0
for q, want in QUERIES:
    hits = vector_store.search(q, top_k=3, min_score=0.0)
    ids = [h["doc_id"] for h in hits]
    top1 += ids[:1] == [want]
    top3 += want in ids
print(f"queries: {len(QUERIES)}  top-1: {top1}/{len(QUERIES)}  top-3: {top3}/{len(QUERIES)}")
