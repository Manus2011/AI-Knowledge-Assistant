# rough latency check for the search path with a few hundred chunks. temp db.
import os, sys, tempfile, time, statistics
tmp = tempfile.mkdtemp()
os.environ["DATA_DIR"] = tmp
os.environ["DB_PATH"] = os.path.join(tmp, "bench.db")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import init_db, get_conn
from app.services import vector_store

init_db()
conn = get_conn()
text = " ".join(f"Policy item {i}: employees must follow procedure {i} for the {['travel','security','vacation','expense'][i % 4]} process." for i in range(40))
for n in range(25):
    conn.execute("INSERT INTO documents (id, filename, uploaded_at) VALUES (?, ?, 'x')", (f"d{n}", f"d{n}.txt"))
conn.commit(); conn.close()
total = sum(vector_store.index_document(f"d{n}", text) for n in range(25))
print("chunks indexed:", total)

times = []
for i in range(30):
    t = time.perf_counter()
    vector_store.search(f"travel procedure {i % 5}", top_k=4)
    times.append((time.perf_counter() - t) * 1000)
print(f"search median {statistics.median(times):.1f} ms, p95 {sorted(times)[int(len(times)*.95)-1]:.1f} ms")
