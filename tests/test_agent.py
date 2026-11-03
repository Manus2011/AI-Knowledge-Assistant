import pytest
from app.services.agent_service import rule_route
from app.services.llm_service import llm_service


@pytest.fixture(autouse=True)
def _no_llm():
    llm_service.set_provider(None)


def upload(client, name, body):
    return client.post("/documents/upload", files={"file": (name, body.encode(), "text/plain")}).json()["document"]


@pytest.mark.parametrize("msg,expected", [
    ("delete old_notes.txt", "delete"),
    ("list my documents", "list"),
    ("show me all the files", "list"),
    ("give me stats", "stats"),
    ("what is the vacation policy?", "search"),
    ("thanks", "chat"),
])
def test_rule_routing(msg, expected):
    assert rule_route(msg) == expected


def test_search_tool(client):
    upload(client, "agent_vac.txt", "Employees get 25 days of paid vacation per year. Requests go to a manager.")
    r = client.post("/agent/run", json={"message": "how many vacation days do employees get?"}).json()
    assert r["status"] == "done"
    assert "search_documents" in r["steps"]
    assert "25 days" in r["answer"]


def test_list_tool(client):
    upload(client, "agent_list.txt", "Some text about budgets and quarterly planning for the finance team.")
    r = client.post("/agent/run", json={"message": "list my documents"}).json()
    assert "agent_list.txt" in r["answer"]


def test_delete_requires_approval_then_executes(client):
    doc = upload(client, "agent_del.txt", "This file will be removed by the agent after approval is given.")
    r = client.post("/agent/run", json={"message": "delete agent_del.txt"}).json()
    assert r["status"] == "awaiting_approval"
    assert client.get(f"/documents/{doc['id']}").status_code == 200  # not deleted yet
    r2 = client.post("/agent/approve", json={"thread_id": r["thread_id"], "approved": True}).json()
    assert r2["status"] == "done" and "Deleted" in r2["answer"]
    assert client.get(f"/documents/{doc['id']}").status_code == 404


def test_delete_rejected_keeps_document(client):
    doc = upload(client, "agent_keep.txt", "Keep this file because the user will deny the deletion request.")
    r = client.post("/agent/run", json={"message": "please remove agent_keep.txt"}).json()
    r2 = client.post("/agent/approve", json={"thread_id": r["thread_id"], "approved": False}).json()
    assert "didn't delete" in r2["answer"]
    assert client.get(f"/documents/{doc['id']}").status_code == 200


def test_delete_unknown_file(client):
    r = client.post("/agent/run", json={"message": "delete nothere.txt"}).json()
    assert r["status"] == "done" and "couldn't tell" in r["answer"]


def test_approve_without_pending_run(client):
    r = client.post("/agent/run", json={"message": "list my documents"}).json()
    assert client.post("/agent/approve", json={"thread_id": r["thread_id"], "approved": True}).status_code == 409
