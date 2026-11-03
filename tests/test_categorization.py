import pytest
from app.services.categorization_service import CategorizationService

CASES = [
    ("Attendees: John and Sarah. Action items assigned to team leads, next steps below.", "meeting_notes"),
    ("This agreement may be terminated with 30 days written notice by either party.", "contract"),
    ("Skills include Python and SQL. Education: BS in Computer Science. Work experience at a startup.", "resume"),
    ("Executive summary: revenue increased 12 percent. Methodology and recommendations follow.", "report"),
]


@pytest.mark.parametrize("backend", ["logreg", "mlp"])
def test_backends_categorize(backend):
    svc = CategorizationService(backend=backend)
    for text, expected in CASES:
        assert svc.categorize(text) == expected


def test_empty_text_is_other():
    assert CategorizationService().categorize("   ") == "other"
