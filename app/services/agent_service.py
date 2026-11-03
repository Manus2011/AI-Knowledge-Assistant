import json
import re
import uuid
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from app.services import agent_tools as tools
from app.services import rag_service
from app.services.llm_service import llm_service

ROUTER_PROMPT = (
    "Pick the one tool that best handles the user's message. Reply with JSON only, like "
    '{"tool": "search"}. Tools: search (questions about document contents), list (list uploaded '
    "documents), stats (document statistics), delete (delete a document), chat (anything else)."
)


class AgentState(TypedDict, total=False):
    message: str
    owner: str
    tool: str
    target_id: str
    target_name: str
    approved: bool
    answer: str
    sources: list
    steps: list


def rule_route(message: str) -> str:
    m = message.lower()
    if re.search(r"\b(delete|remove)\b", m):
        return "delete"
    if re.search(r"\b(list|show)\b.*\b(documents|files|docs)\b", m) or "what documents" in m:
        return "list"
    if re.search(r"\b(stats|statistics|how many documents|how many files)\b", m):
        return "stats"
    if "?" in m or re.search(r"\b(what|who|when|where|how|find|search|explain|summarize)\b", m):
        return "search"
    return "chat"


def route_node(state: AgentState) -> AgentState:
    tool = None
    if llm_service.provider.name == "anthropic":
        # let the model pick, fall back to rules if the output isn't usable json
        try:
            raw = llm_service.provider.generate(ROUTER_PROMPT, [{"role": "user", "content": state["message"]}], 50)
            tool = json.loads(raw[raw.index("{"): raw.rindex("}") + 1]).get("tool")
        except Exception:
            tool = None
    if tool not in {"search", "list", "stats", "delete", "chat"}:
        tool = rule_route(state["message"])
    return {"tool": tool, "steps": [f"route -> {tool}"]}


def pick_branch(state: AgentState) -> str:
    return state["tool"]


def search_node(state: AgentState) -> AgentState:
    result = rag_service.answer(state["message"], owner=state.get("owner"))
    return {"answer": result["answer"], "sources": result["sources"],
            "steps": state["steps"] + ["search_documents"]}


def list_node(state: AgentState) -> AgentState:
    docs = tools.list_documents(state.get("owner"))["documents"]
    if not docs:
        text = "There are no documents uploaded yet."
    else:
        text = "Documents: " + ", ".join(f"{d['filename']} ({d['category']})" for d in docs)
    return {"answer": text, "steps": state["steps"] + ["list_documents"]}


def stats_node(state: AgentState) -> AgentState:
    s = tools.document_stats(state.get("owner"))
    return {"answer": f"{s.get('total_documents', 0)} documents. Details: {json.dumps(s)}",
            "steps": state["steps"] + ["document_stats"]}


def chat_node(state: AgentState) -> AgentState:
    return {"answer": llm_service.chat(state["message"]), "steps": state["steps"] + ["chat"]}


def delete_prepare_node(state: AgentState) -> AgentState:
    doc = tools.find_document(state["message"], state.get("owner"))
    if doc is None:
        return {"target_id": "", "answer": "I couldn't tell which document you want deleted. "
                "Mention the filename.", "steps": state["steps"] + ["delete: no match"]}
    return {"target_id": doc.id, "target_name": doc.filename,
            "steps": state["steps"] + [f"delete: found {doc.filename}"]}


def after_prepare(state: AgentState) -> str:
    return "approve" if state.get("target_id") else "end"


def approval_node(state: AgentState) -> AgentState:
    # pauses the graph here. the api returns the pending action and the run resumes
    # when someone calls /agent/approve with a decision
    decision = interrupt({
        "action": "delete_document",
        "document": state["target_name"],
        "question": f"Delete {state['target_name']}? This can't be undone.",
    })
    return {"approved": bool(decision)}


def after_approval(state: AgentState) -> str:
    return "execute" if state.get("approved") else "cancel"


def execute_delete_node(state: AgentState) -> AgentState:
    tools.delete_document(state["target_id"])
    return {"answer": f"Deleted {state['target_name']}.", "steps": state["steps"] + ["delete_document"]}


def cancel_node(state: AgentState) -> AgentState:
    return {"answer": f"Okay, I didn't delete {state['target_name']}.", "steps": state["steps"] + ["cancelled"]}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("search", search_node)
    g.add_node("list", list_node)
    g.add_node("stats", stats_node)
    g.add_node("chat", chat_node)
    g.add_node("delete", delete_prepare_node)
    g.add_node("approve", approval_node)
    g.add_node("execute", execute_delete_node)
    g.add_node("cancel", cancel_node)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", pick_branch,
                            {"search": "search", "list": "list", "stats": "stats",
                             "chat": "chat", "delete": "delete"})
    g.add_conditional_edges("delete", after_prepare, {"approve": "approve", "end": END})
    g.add_conditional_edges("approve", after_approval, {"execute": "execute", "cancel": "cancel"})
    for n in ("search", "list", "stats", "chat", "execute", "cancel"):
        g.add_edge(n, END)
    return g.compile(checkpointer=MemorySaver())


class AgentService:
    def __init__(self):
        self.graph = build_graph()

    def _format(self, thread_id: str, result: dict) -> dict:
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            return {"status": "awaiting_approval", "thread_id": thread_id, "pending_action": payload}
        return {"status": "done", "thread_id": thread_id, "answer": result.get("answer"),
                "sources": result.get("sources", []), "steps": result.get("steps", [])}

    def run(self, message: str, owner: str, thread_id: str | None = None) -> dict:
        # threads are namespaced by user so one account can't approve another's action
        thread_id = thread_id or str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": f"{owner}:{thread_id}"}}
        result = self.graph.invoke({"message": message, "owner": owner, "steps": []}, cfg)
        return self._format(thread_id, result)

    def resume(self, thread_id: str, approved: bool, owner: str) -> dict:
        cfg = {"configurable": {"thread_id": f"{owner}:{thread_id}"}}
        state = self.graph.get_state(cfg)
        if not state.next:
            raise ValueError("nothing is waiting for approval on this thread")
        result = self.graph.invoke(Command(resume=approved), cfg)
        return self._format(thread_id, result)


agent_service = AgentService()
