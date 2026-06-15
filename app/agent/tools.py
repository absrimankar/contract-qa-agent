from app.vectorstore.store import VectorStore

# ---------------------------------------------------------------------------
# Clause-type keyword registry
# ---------------------------------------------------------------------------

CLAUSE_KEYWORDS: dict[str, list[str]] = {
    "termination": ["terminat", "cancel", "cessation", "end of agreement", "expir"],
    "payment": ["payment", "invoice", "fee", "compensation", "remunerat", "price", "cost", "salary"],
    "liability": ["liabilit", "liable", "indemnif", "damages", "hold harmless", "limitation of liability"],
    "confidentiality": ["confidential", "non-disclosure", "nda", "proprietary information", "trade secret"],
    "intellectual_property": ["intellectual property", "copyright", "patent", "trademark", "ip rights", "work product"],
    "dispute_resolution": ["dispute", "arbitration", "mediation", "governing law", "jurisdiction", "litigation"],
    "force_majeure": ["force majeure", "act of god", "unforeseen circumstance", "beyond.*control"],
    "warranty": ["warrant", "representation", "guarantee", "covenant", "as-is"],
    "renewal": ["renewal", "renew", "extension", "extend", "automatic renewal", "evergreen"],
    "notice": ["notice", "notification", "notify", "written notice", "days' notice"],
}

# ---------------------------------------------------------------------------
# Tool definitions (Anthropic tool-use schema)
# ---------------------------------------------------------------------------


def get_tool_definitions() -> list[dict]:
    return [
        {
            "name": "semantic_search",
            "description": (
                "Search the contract for clauses or text relevant to a natural language query. "
                "Use this to find content related to a concept, obligation, right, or condition "
                "when you do not know the exact clause name."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what you are looking for in the contract.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default 4, max 8).",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "clause_lookup",
            "description": (
                "Find all contract chunks that belong to a known clause type using keyword matching. "
                "Supported types: termination, payment, liability, confidentiality, "
                "intellectual_property, dispute_resolution, force_majeure, warranty, renewal, notice. "
                "Use this first when the user asks about a named clause type."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "clause_type": {
                        "type": "string",
                        "description": (
                            "Clause category to look up. Must be one of: "
                            "termination, payment, liability, confidentiality, "
                            "intellectual_property, dispute_resolution, force_majeure, "
                            "warranty, renewal, notice."
                        ),
                    }
                },
                "required": ["clause_type"],
            },
        },
    ]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _run_semantic_search(store: VectorStore, query: str, k: int = 4) -> str:
    k = min(k, 8)
    results = store.similarity_search(query, k=k)
    if not results:
        return "No relevant clauses found for that query."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Result {i} | Page {r['page_num']} | Score {r['score']:.3f}]\n{r['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _run_clause_lookup(store: VectorStore, clause_type: str) -> str:
    key = clause_type.lower().strip().replace(" ", "_")
    keywords = CLAUSE_KEYWORDS.get(key)

    if keywords is None:
        available = ", ".join(CLAUSE_KEYWORDS.keys())
        return f"Unknown clause type '{clause_type}'. Available types: {available}"

    import re

    matches = []
    for chunk in store.chunks:
        text_lower = chunk["text"].lower()
        if any(re.search(kw, text_lower) for kw in keywords):
            matches.append(chunk)

    if not matches:
        return f"No '{clause_type}' clauses found in the contract."

    parts = []
    for i, chunk in enumerate(matches[:4], 1):
        parts.append(f"[{key.upper()} CLAUSE {i} | Page {chunk['page_num']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def get_tool_executor(store: VectorStore):
    """Return a callable that dispatches tool calls by name."""

    def execute(tool_name: str, tool_input: dict) -> str:
        if tool_name == "semantic_search":
            return _run_semantic_search(store, tool_input["query"], tool_input.get("k", 4))
        if tool_name == "clause_lookup":
            return _run_clause_lookup(store, tool_input["clause_type"])
        return f"Unknown tool: '{tool_name}'"

    return execute
