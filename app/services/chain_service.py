# langchain version of the document chat: prompt | llm | parser, with memory kept in sqlite.
# uses langchain_core only. the "llm" is our llm_service wrapped as a runnable, so it
# uses anthropic with a key and the extractive fallback without one.
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.core import config
from app.services import conversation_service as conv
from app.services import vector_store
from app.services.llm_service import SYSTEM_PROMPT, llm_service

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "<context>\n{context}\n</context>\n\nQuestion: {question}"),
])


class SqliteHistory(BaseChatMessageHistory):
    def __init__(self, key: str):
        self.key = key

    @property
    def messages(self) -> list[BaseMessage]:
        out = []
        for m in conv.get_history(self.key):
            out.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))
        return out

    def add_message(self, message: BaseMessage) -> None:
        conv.add_message(self.key, "user" if message.type == "human" else "assistant", message.content)

    def clear(self) -> None:
        pass


def _call_llm(prompt_value) -> str:
    msgs = prompt_value.to_messages()
    system = msgs[0].content
    chat = [{"role": "user" if m.type == "human" else "assistant", "content": m.content} for m in msgs[1:]]
    return llm_service.provider.generate(system, chat)


# chain: prompt -> llm -> string
_chain = PROMPT | RunnableLambda(_call_llm) | StrOutputParser()


def _with_memory():
    return RunnableWithMessageHistory(
        _chain,
        lambda session_id: SqliteHistory(session_id),
        input_messages_key="question",
        history_messages_key="history",
    )


def ask(question: str, key: str, owner: str | None) -> dict:
    hits = vector_store.search(question, top_k=config.TOP_K, owner=owner)
    if not hits:
        return {"answer": "I couldn't find anything in your documents that answers that.", "sources": []}
    context = "\n\n".join(f"[{i + 1}] ({h['filename']}) {h['text']}" for i, h in enumerate(hits))
    answer = _with_memory().invoke(
        {"question": question, "context": context},
        config={"configurable": {"session_id": key}},
    )
    return {"answer": answer, "sources": [{"ref": i + 1, "filename": h["filename"], "score": h["score"]}
                                          for i, h in enumerate(hits)]}
