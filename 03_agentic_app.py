"""
UDA-Hub demo entry point.

HOW TO RUN
----------
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your OPENAI_API_KEY.
3. Seed the databases (only needed once):
       jupyter nbconvert --to notebook --execute 01_external_db_setup.ipynb
       jupyter nbconvert --to notebook --execute 02_core_db_setup.ipynb
   (or just open and "Run All" in each notebook)
4. `python 03_agentic_app.py`
   - runs 6 scripted demo tickets end-to-end (classification -> routing ->
     retrieval/tools -> resolution or escalation -> memory write), printing
     each step's decision.
   - then drops you into the interactive chat_interface() REPL from utils.py
     so you can try your own tickets. Ctrl+C or type "exit" to quit before
     the interactive part if you only want the scripted demo.

You can also `from agentic.workflow import run_ticket` and call it directly
from a notebook (see 03_agentic_app.ipynb-equivalent usage in the README).
"""
from __future__ import annotations

import sys

import config
from agentic.workflow import run_ticket
from utils import chat_interface

DEMO_TICKETS = [
    dict(
        label="A) Straightforward knowledge-base resolution (no tools, no account needed)",
        subject="How do I change notification settings?",
        description="Hi, I'm getting way too many push notifications about new experiences. "
        "How do I turn those off but keep booking reminders?",
        channel="email",
        metadata={},
    ),
    dict(
        label="B) Refund WITHIN the eligibility window -> resolver uses account_lookup_tool + refund_tool",
        subject="Refund for Planetarium Night Show",
        description="I booked the Planetarium Night Show but I can't make it anymore, can I get a refund?",
        channel="chat",
        metadata={"email": "jordan.blake@example.com"},
    ),
    dict(
        label="C) Refund OUTSIDE the eligibility window -> refund_tool reports not eligible -> escalate",
        subject="Duplicate charge for Sculpture Garden Tour",
        description="I was charged twice for the Sculpture Garden Tour booking a while back — "
        "please refund the duplicate charge.",
        channel="email",
        metadata={"email": "priya.shah@example.com"},
    ),
    dict(
        label="D) Critical urgency -> Supervisor escalates immediately, before Resolver even runs",
        subject="Locked out at the door, event starting now",
        description="URGENT — I'm standing outside the venue right now and my QR code won't scan, "
        "the show starts in 5 minutes and staff won't let me in!",
        channel="chat",
        metadata={"email": "lan.nguyen@example.com"},
    ),
    dict(
        label="E) Off-topic / no matching knowledge article -> low RAG confidence -> escalate",
        subject="Birthday party planning",
        description="Can your team help me plan and host a surprise 40th birthday party at one of "
        "your partner venues, catering included?",
        channel="chat",
        metadata={"email": "omar.haddad@example.com"},
    ),
]


def run_demo() -> None:
    if not config.OPENAI_API_KEY:
        print(
            "ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 100)
    print("UDA-HUB SCRIPTED DEMO — 5 tickets covering resolution, tool use, and 3 kinds of escalation")
    print("=" * 100)

    last_thread_for_memory_demo = None
    for i, ticket in enumerate(DEMO_TICKETS, start=1):
        print(f"\n\n----- Ticket {i}: {ticket['label']} -----")
        print(f"Customer ({ticket['metadata'].get('email', 'anonymous')}): {ticket['description']}")
        state = run_ticket(
            subject=ticket["subject"],
            description=ticket["description"],
            channel=ticket["channel"],
            metadata=ticket["metadata"],
        )
        print(f"\n>>> status: {state['status']}  |  ticket_id: {state.get('ticket_id')}  |  "
              f"classification: {state.get('classification')}")
        if state["status"] == "escalated":
            print(f">>> escalation note for human agent:\n{state.get('escalation_summary')}")
        else:
            print(f">>> resolution sent to customer:\n{state.get('resolution')}")
        if ticket["metadata"].get("email") == "lan.nguyen@example.com":
            last_thread_for_memory_demo = ticket["metadata"]["email"]

    print("\n\n----- Ticket 6: F) Follow-up ticket, SAME customer as Ticket 4, NEW session -----")
    print("Demonstrates long-term memory recall across sessions (Supervisor pulls prior resolution "
          "summaries for this customer even though this is a brand-new thread_id).")
    state = run_ticket(
        subject="Following up on my access issue",
        description="Following up on the door access problem I had recently — did that ever get resolved "
        "on your end, and is there anything I should do differently next time?",
        channel="chat",
        metadata={"email": "lan.nguyen@example.com"},
    )
    print(f"\n>>> status: {state['status']}  |  long_term_context recalled: {state.get('long_term_context')}")

    print("\n\n" + "=" * 100)
    print("Full structured logs for every ticket above are in data/core/udahub.db -> agent_run_log")
    print("=" * 100)


if __name__ == "__main__":
    run_demo()
    print("\n\nDemo complete. Starting interactive chat — type 'exit' anytime to quit.\n")
    try:
        chat_interface()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
