"""
UDA-Hub tools — plain LangChain `@tool` functions that abstract all database
access (both the external CultPass DB and UDA-Hub's own core DB) away from
the agents. Agents never touch SQLAlchemy directly; they only ever call these.

Each tool is a pure function of (typed args) -> (small JSON-serializable
dict), with no shared state between calls, which is what makes it mechanical
to later re-host these behind a FastMCP server if UDA-Hub ever needs to expose
them to a *different* process (e.g. a separate resolver microservice): wrap
each function body in an `@mcp.tool()` decorator instead of `@tool`, and swap
the in-process import in agentic/workflow.py for an MCP client call. Nothing
about the function signatures or return shapes would need to change.
"""
