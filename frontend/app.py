"""
Standalone Streamlit app — no FastAPI backend required.
Imports the ingestion pipeline, vector store, and LangGraph agent directly.
Designed for Streamlit Cloud: set ANTHROPIC_API_KEY in App Secrets.
"""

import os
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Bootstrap: sys.path + secrets injection
# Must happen before any `app.*` imports so pydantic-settings sees the key.
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Streamlit Cloud secrets → os.environ (pydantic-settings reads from env)
if hasattr(st, "secrets"):
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)

# ---------------------------------------------------------------------------
# App imports (safe now that env is populated)
# ---------------------------------------------------------------------------

from app.agent.graph import create_agent, run_agent  # noqa: E402
from app.ingestion.chunker import chunk_pages  # noqa: E402
from app.ingestion.embedder import build_index  # noqa: E402
from app.ingestion.pdf_parser import parse_pdf  # noqa: E402
from app.vectorstore.store import VectorStore  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Contract Q&A Agent",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "contract_loaded" not in st.session_state:
    st.session_state.contract_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []
if "agent" not in st.session_state:
    st.session_state.agent = None

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def process_contract(file_bytes: bytes) -> tuple[int, int]:
    """Run the full ingestion pipeline and build a cached agent."""
    pages = parse_pdf(file_bytes)
    if not pages:
        raise ValueError(
            "No extractable text found in this PDF. "
            "Scanned / image-only PDFs are not supported."
        )
    chunks = chunk_pages(pages)
    build_index(chunks)

    store = VectorStore()
    store.load()

    st.session_state.agent = create_agent(store)
    st.session_state.contract_loaded = True
    st.session_state.chat_history = []

    return len(pages), len(chunks)


# ---------------------------------------------------------------------------
# Sidebar — contract upload
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📄 Contract Q&A Agent")
    st.markdown("---")
    st.subheader("1. Upload Contract")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        if st.button("Process Contract", type="primary", use_container_width=True):
            with st.spinner("Parsing and indexing contract…"):
                try:
                    page_count, chunk_count = process_contract(uploaded_file.getvalue())
                    st.success("Contract uploaded and indexed successfully.")
                    st.info(
                        f"**Pages indexed:** {page_count}  \n"
                        f"**Chunks created:** {chunk_count}"
                    )
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    st.markdown("---")
    st.subheader("2. Ask Questions")
    st.markdown(
        "After uploading, type questions in the chat window.  \n\n"
        "**Example questions:**\n"
        "- What are the termination conditions?\n"
        "- How are disputes resolved?\n"
        "- What does the contract say about liability limits?\n"
        "- When are payments due?"
    )

    if st.session_state.contract_loaded:
        st.success("Contract ready — ask away!")
    else:
        st.warning("No contract loaded yet.")

# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------

st.header("Chat with your Contract")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the contract…"):
    if not st.session_state.contract_loaded:
        st.warning("Please upload and process a contract first (sidebar).")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing contract…"):
            try:
                answer = run_agent(st.session_state.agent, prompt)
                st.markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                error_msg = f"Agent error: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"**{error_msg}**"}
                )
