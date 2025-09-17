# app/ui.py

import streamlit as st
from app.services.rag_pipeline import ask_question, qa_chain

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="🩺 Smart Medical Chatbot", layout="wide")
st.title("🩺 Smart Medical Chatbot (RAG + Memory)")

# -----------------------
# Session state defaults
# -----------------------
if "last_sources" not in st.session_state:
    st.session_state["last_sources"] = []
if "buffer_input" not in st.session_state:  # buffer for user input
    st.session_state["buffer_input"] = ""

# -----------------------
# Handle sending message
# -----------------------
def handle_send():
    query = st.session_state["buffer_input"].strip()
    if not query:
        st.warning("Please enter a valid question.")
        return

    with st.spinner("Generating answer..."):
        try:
            answer, sources = ask_question(query)
        except Exception as e:
            st.error(f"Error while generating answer: {e}")
            answer, sources = "Error generating answer.", []

    st.session_state["last_sources"].append(
        {"query": query, "answer": answer, "sources": sources}
    )

    # clear the buffer safely
    st.session_state["buffer_input"] = ""


# -----------------------
# Input area
# -----------------------
st.text_area(
    "Ask your medical question:",
    height=120,
    key="buffer_input"
)

col1, col2 = st.columns([1, 1])

with col1:
    st.button("Send", on_click=handle_send)

with col2:
    if st.button("Reset Conversation"):
        # Clear LangChain memory
        if hasattr(qa_chain, "memory") and qa_chain.memory is not None:
            try:
                qa_chain.memory.clear()
            except Exception:
                try:
                    qa_chain.memory.reset()
                except Exception:
                    pass
        st.session_state["last_sources"] = []
        st.session_state["buffer_input"] = ""
        st.success("Conversation reset successfully.")

# -----------------------
# Display conversation from memory
# -----------------------
def render_message(msg):
    role = getattr(msg, "type", None) or getattr(msg, "role", None)
    text = getattr(msg, "content", None) or getattr(msg, "text", None)
    if not text:
        text = str(msg)
    if role and role.lower() in ("human", "user"):
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")
    st.markdown("---")

if hasattr(qa_chain, "memory") and qa_chain.memory is not None:
    try:
        mem = qa_chain.memory.load_memory_variables({})
        history = mem.get("chat_history", [])
    except Exception:
        history = []
    if history:
        st.subheader("Conversation")
        for msg in history:
            render_message(msg)

# -----------------------
# Show sources
# -----------------------
if st.checkbox("Show Source Documents"):
    st.subheader("Retrieved Sources")
    for entry in reversed(st.session_state["last_sources"]):
        st.markdown(f"**Q:** {entry['query']}")
        st.markdown(f"**A:** {entry['answer']}")
        for i, doc in enumerate(entry.get("sources", []), start=1):
            content = getattr(doc, "page_content", None) or str(doc)
            meta = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
            src = meta.get("source") or "N/A"
            st.markdown(f"**Source {i}:** {src}")
            st.markdown(content[:500] + ("..." if len(content) > 500 else ""))
        st.markdown("---")