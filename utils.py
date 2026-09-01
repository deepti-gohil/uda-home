"""Simple terminal chat loop for driving the UDA-Hub agentic workflow by hand."""
from __future__ import annotations

from agentic.workflow import run_ticket


def _print_outcome(state: dict) -> None:
    print(f"\n[thread_id: {state['thread_id']}  |  ticket_id: {state.get('ticket_id')}  |  "
          f"status: {state.get('status')}]")
    classification = state.get("classification") or {}
    if classification:
        print(
            f"classification: category={classification.get('category')} "
            f"urgency={classification.get('urgency')} sentiment={classification.get('sentiment')}"
        )
    if state.get("status") == "escalated":
        print(f"\n[ESCALATED to a human agent]\n{state.get('escalation_summary')}")
    else:
        print(f"\nAgent: {state.get('resolution')}")


def chat_interface() -> None:
    """A minimal REPL: each conversation stays on one ticket/thread until you
    type 'new' to start another, or 'exit'/'quit' to stop. Optionally start by
    entering a CultPass email so the agent can look up your account and
    long-term memory (press Enter to skip)."""
    print("=== UDA-Hub chat (CultPass support) ===")
    print("Type 'new' to start a fresh ticket, 'exit' or 'quit' to leave.\n")

    email = input("Your CultPass email (optional, press Enter to skip): ").strip() or None
    thread_id: str | None = None

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if user_input.lower() == "new":
            thread_id = None
            print("--- started a new ticket/session ---")
            continue

        subject = user_input[:80]
        state = run_ticket(
            subject=subject,
            description=user_input,
            channel="chat",
            metadata={"email": email} if email else {},
            thread_id=thread_id,
        )
        thread_id = state["thread_id"]
        _print_outcome(state)


if __name__ == "__main__":
    chat_interface()
