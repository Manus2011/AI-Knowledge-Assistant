import re
from app.core import config

SYSTEM_PROMPT = (
    "You are an assistant for an internal document knowledge base. "
    "Answer briefly and clearly. If you are given document context, only use that context "
    "and say so when it doesn't contain the answer."
)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system: str, messages: list[dict], max_tokens: int = 600) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class ExtractiveProvider:
    # used when there is no api key. doesn't generate anything, it just pulls the
    # sentences from the context that overlap most with the question
    name = "extractive"

    def generate(self, system: str, messages: list[dict], max_tokens: int = 600) -> str:
        last = messages[-1]["content"]
        context, question = split_context(last)
        if not context:
            return ("No LLM api key is configured, so I can't write free-form answers. "
                    "Set ANTHROPIC_API_KEY, or ask a question about your uploaded documents "
                    "and I'll pull the most relevant passages.")
        sentences = re.split(r"(?<=[.!?])\s+", context)
        q_words = set(re.findall(r"[a-z]{3,}", question.lower()))
        scored = []
        for s in sentences:
            overlap = len(q_words & set(re.findall(r"[a-z]{3,}", s.lower())))
            if overlap:
                scored.append((overlap, s))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return "I couldn't find anything in the documents that answers that."
        return " ".join(s for _, s in scored[:2])


def build_prompt(question: str, context: str | None = None) -> str:
    if not context:
        return question
    return f"<context>\n{context}\n</context>\n\nQuestion: {question}"


def split_context(prompt: str) -> tuple[str, str]:
    m = re.search(r"<context>\n(.*?)\n</context>\n\nQuestion: (.*)", prompt, re.S)
    if not m:
        return "", prompt
    return m.group(1), m.group(2)


class LLMService:
    def __init__(self):
        self._provider = None

    @property
    def provider(self):
        if self._provider is None:
            if config.ANTHROPIC_API_KEY:
                self._provider = AnthropicProvider(config.ANTHROPIC_API_KEY, config.LLM_MODEL)
            else:
                self._provider = ExtractiveProvider()
        return self._provider

    def set_provider(self, provider):
        # tests swap in a fake one
        self._provider = provider

    def chat(self, question: str, history: list[dict] | None = None,
             context: str | None = None, system: str = SYSTEM_PROMPT) -> str:
        messages = list(history or [])
        messages.append({"role": "user", "content": build_prompt(question, context)})
        return self.provider.generate(system, messages)


llm_service = LLMService()
