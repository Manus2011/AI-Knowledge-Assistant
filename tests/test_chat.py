from app.services.llm_service import llm_service, ExtractiveProvider, build_prompt


class FakeProvider:
    name = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, system, messages, max_tokens=600):
        self.calls.append(messages)
        return f"echo {len(messages)}"


def test_fallback_without_key(client):
    llm_service.set_provider(None)
    r = client.post("/chat/", json={"message": "hello"})
    assert r.status_code == 200
    assert r.json()["provider"] == "extractive"


def test_memory_across_turns(client):
    fake = FakeProvider()
    llm_service.set_provider(fake)
    r1 = client.post("/chat/", json={"message": "first"}).json()
    r2 = client.post("/chat/", json={"message": "second", "conversation_id": r1["conversation_id"]}).json()
    assert r1["reply"] == "echo 1"
    assert r2["reply"] == "echo 3"  # first user + first reply + new question
    hist = client.get(f"/chat/{r1['conversation_id']}").json()["messages"]
    assert [m["role"] for m in hist] == ["user", "assistant", "user", "assistant"]
    llm_service.set_provider(None)


def test_empty_message_rejected(client):
    assert client.post("/chat/", json={"message": ""}).status_code == 422


def test_extractive_picks_relevant_sentence():
    p = ExtractiveProvider()
    ctx = "The launch was delayed two weeks. Sarah owns the QA checklist. Lunch is at noon."
    out = p.generate("", [{"role": "user", "content": build_prompt("Who owns the QA checklist?", ctx)}])
    assert "Sarah" in out and "Lunch" not in out


def test_anthropic_provider_wiring(monkeypatch):
    # no network: check we pass system + messages through to the sdk and join text blocks
    from app.services.llm_service import AnthropicProvider

    class Block:
        type = "text"
        def __init__(self, t): self.text = t

    class Resp:
        content = [Block("hi "), Block("there")]

    class Messages:
        def create(self, **kw):
            Messages.kw = kw
            return Resp()

    class Client:
        messages = Messages()

    p = AnthropicProvider.__new__(AnthropicProvider)
    p.client, p.model = Client(), "m"
    assert p.generate("sys", [{"role": "user", "content": "q"}]) == "hi there"
    assert Messages.kw["system"] == "sys" and Messages.kw["model"] == "m"
