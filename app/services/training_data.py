# builds a bigger synthetic training set by mixing sentence fragments per category.
# it's templated so accuracy on it is optimistic, real docs would be messier.
import random

FRAGMENTS = {
    "resume": [
        "Experienced software engineer with {n} years in Python and backend development.",
        "Skills include SQL, Excel, Tableau and machine learning.",
        "Education: {deg} in Computer Science from a state university.",
        "Work experience includes internships at two startups and a consulting firm.",
        "Objective: seeking a data analyst role where I can apply my analytical skills.",
        "Led a team of {n} developers and delivered projects ahead of schedule.",
        "Certifications: AWS Cloud Practitioner, Google Data Analytics.",
        "References available upon request.",
    ],
    "meeting_notes": [
        "Meeting notes from {day}: discussed the Q{n} roadmap and open blockers.",
        "Attendees: John, Sarah, Mike and Priya.",
        "Action items assigned to team leads, follow up next week.",
        "Agenda: standup updates, budget review, upcoming deadlines.",
        "Next steps: schedule a review session and share the minutes with the group.",
        "Decisions made: postpone the launch by {n} weeks, revisit after testing.",
        "Discussion points included hiring, vendor contracts and sprint planning.",
        "Meeting adjourned at 3pm, next sync on {day}.",
    ],
    "report": [
        "Quarterly report summarizing revenue growth and key performance indicators.",
        "This report outlines findings from the user research study conducted in June.",
        "Executive summary: revenue increased {n} percent compared to the previous year.",
        "Methodology: data was collected from surveys and analyzed using regression.",
        "Annual report detailing company performance across all business units.",
        "Figure {n} shows the trend in customer retention over the last four quarters.",
        "Conclusion and recommendations: invest further in the highest performing segments.",
        "Appendix contains the full data tables and statistical analysis.",
    ],
    "contract": [
        "This agreement is entered into between the parties for the purpose of service delivery.",
        "Terms and conditions: payment due within {n} days of invoice date.",
        "Non-disclosure agreement covering confidential information shared between parties.",
        "The client shall indemnify the provider against any third party claims.",
        "This contract may be terminated by either party with {n} days written notice.",
        "Governing law: this agreement is governed by the laws of the State of New York.",
        "Liability is limited to the total fees paid under this agreement.",
        "Signed by the authorized representatives of both parties on the date above.",
    ],
}


def _fill(template, rng):
    return (template
            .replace("{n}", str(rng.randint(2, 30)))
            .replace("{deg}", rng.choice(["BS", "MS", "BA"]))
            .replace("{day}", rng.choice(["Monday", "Tuesday", "Friday", "March 4", "June 12"])))


def generate(n_per_class: int, seed: int, frag_slice: slice = slice(None)):
    # frag_slice lets you hold out some fragments completely for a harder test
    rng = random.Random(seed)
    data = []
    for label, frags in FRAGMENTS.items():
        frags = frags[frag_slice]
        for _ in range(n_per_class):
            picks = rng.sample(frags, rng.randint(2, min(4, len(frags))))
            data.append((" ".join(_fill(p, rng) for p in picks), label))
    rng.shuffle(data)
    return data
