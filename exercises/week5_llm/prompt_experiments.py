# tries a few prompt styles on the same task. needs ANTHROPIC_API_KEY, otherwise it
# just prints the prompts so you can see what would be sent.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.services.llm_service import llm_service

DOC = ("Meeting notes: the team agreed to delay the launch by two weeks. "
       "Sarah owns the QA checklist. Mike will update the vendor contract by Friday.")

PROMPTS = {
    "zero_shot": f"Who owns what in this note?\n\n{DOC}",
    "structured": f"List each person and their task as 'name: task', one per line, nothing else.\n\n{DOC}",
    "few_shot": (
        "Extract owners.\n\nNote: Priya will send the report. Tom books the room.\n"
        "Owners:\nPriya: send the report\nTom: book the room\n\n"
        f"Note: {DOC}\nOwners:"
    ),
}

has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
for name, prompt in PROMPTS.items():
    print(f"=== {name} ===")
    if has_key:
        print(llm_service.chat(prompt, system="You answer concisely."))
    else:
        print(prompt)
    print()
