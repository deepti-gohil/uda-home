# UDA-Hub — Universal Decision Agent for CultPass Support

A LangGraph multi-agent system that reads incoming CultPass support tickets,
classifies and routes them, retrieves knowledge (RAG) or calls tools to
resolve them, escalates when it can't, and remembers customers across
sessions. See [`agentic/design/architecture.md`](agentic/design/architecture.md)
for the full design writeup (architecture diagram, agent roster, routing
rules, RAG design, memory design).

> **Note on the starter code**: this submission was developed without access
> to the original Udacity `starter/` repo (it wasn't available in the local
> environment this was built in), so the whole scaffold — database schema,
> setup notebooks, `cultpass_articles` content, agents, tools, and
> `workflow.py` — was built from scratch against the written project brief
> rather than adapted from provided starter files. Nothing here was copied
> from a `starter/` folder.

## Requirements

- **Python 3.11+** (developed and tested on 3.11)
- An OpenAI API key (used for chat completions and `text-embedding-3-small`
  embeddings — see [`requirements.txt`](requirements.txt) for exact package
  versions: `langgraph`, `langgraph-checkpoint-sqlite`, `langchain`,
  `langchain-openai`, `langchain-community`, `openai`, `faiss-cpu`,
  `sqlalchemy`, `numpy`, `pandas`, `pydantic`, `pytest`, `jupyter`)

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your OPENAI_API_KEY
```

## Setup — run once

Both databases are plain SQLite files under `data/`, created + seeded by two
notebooks:

```bash
jupyter nbconvert --to notebook --execute 01_external_db_setup.ipynb
jupyter nbconvert --to notebook --execute 02_core_db_setup.ipynb
```

(or just open each in Jupyter and "Run All" — both are idempotent, so
re-running is safe and skips re-seeding if data already exists.)

- **`01_external_db_setup.ipynb`** → `data/external/cultpass.db` — CultPass's
  own data: `accounts`, `users`, `bookings`.
- **`02_core_db_setup.ipynb`** → `data/core/udahub.db` — UDA-Hub's own data:
  `tickets`, `ticket_metadata`, `ticket_messages`, `knowledge` (18 seeded
  articles across 10 categories), `long_term_memory`, `agent_run_log`. This
  notebook also builds the FAISS RAG index on first run and sanity-checks
  retrieval.

## Run the app

```bash
python 03_agentic_app.py
```

This runs **6 scripted demo tickets** end-to-end (see the script for the
exact scenarios) that between them exercise: plain knowledge-base
resolution, a tool-driven refund that succeeds, a tool-driven refund that's
correctly rejected as outside policy (and escalates), an immediate
critical-urgency escalation, an off-topic ticket that escalates on low RAG
confidence, and a follow-up session for a repeat customer that demonstrates
long-term memory recall. It then drops you into an interactive chat loop
(`chat_interface()` from `utils.py`) so you can try your own tickets —
type `exit` to quit, `new` to start a fresh ticket in the same run.

You can also drive it directly from a notebook or the REPL:

```python
from agentic.workflow import run_ticket

state = run_ticket(
    subject="Refund request",
    description="I can't make my Planetarium Night Show booking, can I get a refund?",
    channel="chat",
    metadata={"email": "jordan.blake@example.com"},
)
print(state["status"], state["resolution"] or state["escalation_summary"])
```

## Run the tests

```bash
pytest
```

Tests that call OpenAI (RAG retrieval, memory embeddings, full workflow
runs) are automatically skipped if `OPENAI_API_KEY` isn't set, so `pytest`
still runs the DB/tool-abstraction tests without a key. See
`tests/test_tools.py` and `tests/test_workflow.py`.

## Project structure

```
solution/
├── agentic/
│   ├── agents/            classifier, supervisor, resolver, escalation, memory
│   ├── design/
│   │   └── architecture.md   full design doc + Mermaid diagram
│   ├── tools/              kb_search (RAG), account_lookup, refund, memory
│   ├── state.py            shared TicketState (TypedDict)
│   ├── logging_utils.py    structured decision/tool/routing logging
│   └── workflow.py         StateGraph built from scratch (no prebuilt agent)
├── data/
│   ├── core/                UDA-Hub's own DB: models, seed data, db.py
│   ├── external/             CultPass's DB: models, seed data, db.py
│   └── models/               FAISS knowledge index cache (generated)
├── tests/                    pytest suite
├── config.py                  central settings + absolute path resolution
├── utils.py                   chat_interface()
├── 01_external_db_setup.ipynb
├── 02_core_db_setup.ipynb
├── 03_agentic_app.py          demo + interactive entry point
├── requirements.txt
└── .env.example
```

## Design highlights

- **Pattern**: Supervisor (5 specialized agents: Classifier, Supervisor,
  Resolver, Escalation, Memory) — see architecture.md §3/§9 for why.
- **Routing**: deterministic, rule-based in the Supervisor (not an LLM call)
  based on urgency/sentiment/category, with a second confidence-based
  escalation check inside the Resolver after RAG retrieval — architecture.md §7.
- **RAG**: FAISS over OpenAI embeddings of the `knowledge` table, cached to
  disk, rebuilt automatically when the table changes; a 0.72 cosine-similarity
  confidence gate decides resolve-vs-escalate — architecture.md §5.
- **Tools**: `kb_search_tool`, `account_lookup_tool`, `refund_tool`,
  `write_long_term_memory`/`search_long_term_memory` — all plain functions
  with structured returns and input validation, DB paths resolved absolutely
  via `config.py` — architecture.md §6.
- **Short-term memory**: LangGraph `SqliteSaver` checkpointer keyed by
  `thread_id`; `TicketState.messages` uses the `add_messages` reducer.
- **Long-term memory**: a `long_term_memory` table with per-row OpenAI
  embeddings, searched via NumPy cosine similarity, scoped per customer and
  persisted across brand-new sessions — architecture.md §4.
- **Logging**: every node calls `log_event()`, which both prints and writes a
  structured row to `agent_run_log` (queryable with plain SQL) —
  satisfies the "log agent decisions, routing choices, tool usage" requirement.

## What's not implemented (and why)

- **MCP servers for tools**: considered, documented as a natural next step in
  `agentic/tools/__init__.py` and architecture.md §6, but not implemented —
  this runs as a single local process, so a network hop would add complexity
  without changing behavior for this submission.
- **Vector DB for long-term memory**: intentionally NumPy cosine similarity
  instead — per-customer memory counts are small enough that a dedicated
  vector store would be over-engineering here (architecture.md §4).
