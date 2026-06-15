from typing import TypedDict

import anthropic
from langgraph.graph import END, StateGraph

from app.agent.tools import get_tool_definitions, get_tool_executor
from app.core.config import settings
from app.vectorstore.store import VectorStore

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a contract analysis assistant. Answer questions about the uploaded contract accurately and concisely.

Guidelines:
- Always search the contract with the available tools before answering.
- For questions about a named clause type (e.g. termination, payment, liability), call clause_lookup first, then semantic_search for further context.
- Always cite the page number and quote the relevant contract text in your answer.
- Distinguish clearly between what the contract explicitly states and what you infer.
- If the contract does not contain the requested information, say so explicitly — do not speculate."""


class AgentState(TypedDict):
    messages: list[dict]
    final_answer: str


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------


def _serialize_content(content_blocks) -> list[dict]:
    """Convert Anthropic SDK content objects → plain dicts for state storage."""
    out = []
    for block in content_blocks:
        if block.type == "text":
            out.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            out.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def create_agent(store: VectorStore):
    """Build and compile a LangGraph ReAct agent bound to *store*."""
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    tool_defs = get_tool_definitions()
    tool_executor = get_tool_executor(store)

    # --- nodes -----------------------------------------------------------

    def call_llm(state: AgentState) -> dict:
        response = client.messages.create(
            model=settings.LLM_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tool_defs,
            messages=state["messages"],
        )
        assistant_msg = {
            "role": "assistant",
            "content": _serialize_content(response.content),
        }
        return {"messages": state["messages"] + [assistant_msg]}

    def execute_tools(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        tool_results = []
        for block in last_msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                result = tool_executor(block["name"], block["input"])
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": result,
                    }
                )
        tool_msg = {"role": "user", "content": tool_results}
        return {"messages": state["messages"] + [tool_msg]}

    def extract_answer(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        text_parts = [
            b["text"]
            for b in last_msg.get("content", [])
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        answer = "\n".join(text_parts).strip()
        return {"final_answer": answer or "I could not find an answer in the contract."}

    # --- routing ---------------------------------------------------------

    def should_continue(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        for block in last_msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return "execute_tools"
        return "end"

    # --- graph assembly --------------------------------------------------

    graph = StateGraph(AgentState)
    graph.add_node("call_llm", call_llm)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("extract_answer", extract_answer)

    graph.set_entry_point("call_llm")
    graph.add_conditional_edges(
        "call_llm",
        should_continue,
        {"execute_tools": "execute_tools", "end": "extract_answer"},
    )
    graph.add_edge("execute_tools", "call_llm")
    graph.add_edge("extract_answer", END)

    return graph.compile()


def run_agent(agent, question: str) -> str:
    """Invoke the compiled graph with a single user question."""
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": question}],
        "final_answer": "",
    }
    result = agent.invoke(initial_state)
    return result["final_answer"]
