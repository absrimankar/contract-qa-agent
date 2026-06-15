import requests
import streamlit as st

API_BASE = "http://localhost:8000"

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
                    response = requests.post(
                        f"{API_BASE}/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        },
                        timeout=120,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(data["message"])
                        st.info(
                            f"**Pages indexed:** {data['page_count']}  \n"
                            f"**Chunks created:** {data['chunk_count']}"
                        )
                        st.session_state.contract_loaded = True
                        st.session_state.chat_history = []
                    else:
                        detail = response.json().get("detail", "Upload failed.")
                        st.error(f"Error {response.status_code}: {detail}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot reach the API at `localhost:8000`.  \n"
                        "Make sure the backend is running:  \n"
                        "```\nuvicorn app.main:app --reload\n```"
                    )

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

# Render existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the contract…"):
    if not st.session_state.contract_loaded:
        st.warning("Please upload and process a contract first (sidebar).")
        st.stop()

    # Display user message immediately
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query the agent
    with st.chat_message("assistant"):
        with st.spinner("Analysing contract…"):
            try:
                response = requests.post(
                    f"{API_BASE}/query",
                    json={"question": prompt},
                    timeout=120,
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    detail = response.json().get("detail", "Query failed.")
                    error_msg = f"Error {response.status_code}: {detail}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"**{error_msg}**"}
                    )
            except requests.exceptions.ConnectionError:
                msg = "Lost connection to the API. Is the backend still running?"
                st.error(msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"**{msg}**"}
                )
