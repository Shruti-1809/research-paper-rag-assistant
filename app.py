"""
RAG Research Assistant - Streamlit Frontend
Run with: streamlit run app.py
"""

import streamlit as st
import requests
import time

API_BASE = "http://localhost:8000"

# --- Page Config --------------------------------------------------------------

st.set_page_config(
    page_title="Research RAG Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---------------------------------------------------------------

st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
    }
    .citation-box {
        background: #1e2530;
        border-left: 3px solid #4ade80;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 8px;
        font-size: 13px;
    }
    .confidence-text {
        color: #4ade80;
        font-size: 12px;
        font-weight: 500;
    }
    .paper-title {
        font-weight: 600;
        color: #4ade80;
    }
    .chat-user {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 10px 14px;
        margin: 6px 0;
    }
    .chat-assistant {
        background: #0d1117;
        border-radius: 12px;
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 3px solid #059669;
    }
</style>
""", unsafe_allow_html=True)

# --- API Helpers --------------------------------------------------------------

def upload_paper(file):
    try:
        res = requests.post(
            f"{API_BASE}/papers/upload",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            timeout=30,
        )
        if res.status_code == 201:
            return res.json(), None
        return None, res.json().get("detail", "Upload failed")
    except Exception as e:
        return None, str(e)


def get_paper(paper_id):
    try:
        res = requests.get(f"{API_BASE}/papers/{paper_id}", timeout=10)
        return res.json()
    except:
        return None


def list_papers():
    try:
        res = requests.get(f"{API_BASE}/papers/", timeout=10)
        return res.json().get("papers", [])
    except:
        return []


def delete_paper(paper_id):
    try:
        requests.delete(f"{API_BASE}/papers/{paper_id}", timeout=10)
        return True
    except:
        return False


def send_chat(paper_id, message, session_id=None):
    try:
        res = requests.post(
            f"{API_BASE}/chat/",
            json={"paper_id": paper_id, "message": message, "session_id": session_id},
            timeout=60,
        )
        if res.status_code == 200:
            return res.json(), None
        return None, res.json().get("detail", "Chat failed")
    except Exception as e:
        return None, str(e)


def summarize_paper(paper_id):
    try:
        res = requests.post(
            f"{API_BASE}/analysis/summarize",
            json={"paper_id": paper_id},
            timeout=120,
        )
        if res.status_code == 200:
            return res.json(), None
        return None, "Summarize failed"
    except Exception as e:
        return None, str(e)


def check_backend():
    try:
        res = requests.get(f"{API_BASE}/health", timeout=5)
        return res.status_code == 200
    except:
        return False


# --- Session State Init -------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "selected_paper" not in st.session_state:
    st.session_state.selected_paper = None
if "papers" not in st.session_state:
    st.session_state.papers = []


# --- Sidebar ------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔬 Research RAG")
    st.markdown("*AI-powered paper assistant*")
    st.divider()

    # Backend status
    backend_ok = check_backend()
    if backend_ok:
        st.success("✅ Backend connected", icon="🟢")
    else:
        st.error("❌ Backend not running. Start with: `python main.py`")
        st.stop()

    st.divider()

    # Upload section
    st.markdown("### 📄 Upload Paper")
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload a research paper PDF (IEEE, ACM, Springer, etc.)",
    )

    if uploaded_file:
        if st.button("🚀 Upload & Process", type="primary"):
            with st.spinner("Uploading..."):
                paper, err = upload_paper(uploaded_file)
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.success(f"Uploaded! Processing in background...")
                    st.session_state.papers = list_papers()
                    st.session_state.selected_paper = paper
                    st.session_state.messages = []
                    st.session_state.session_id = None
                    st.rerun()

    st.divider()

    # Papers list
    st.markdown("### 📚 Your Papers")

    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.papers = list_papers()
        st.rerun()

    # Load papers if empty
    if not st.session_state.papers:
        st.session_state.papers = list_papers()

    papers = st.session_state.papers

    if not papers:
        st.info("No papers yet. Upload one above.")
    else:
        for paper in papers:
            col1, col2 = st.columns([4, 1])
            with col1:
                title = paper.get("title") or paper.get("filename", "Unknown")
                title_short = title[:35] + "..." if len(title) > 35 else title
                is_processed = paper.get("is_processed", False)
                status = "✅" if is_processed else "⏳"
                chunks = paper.get("chunk_count", 0)

                if st.button(
                    f"{status} {title_short}",
                    key=f"select_{paper['id']}",
                    use_container_width=True,
                    help=f"Chunks: {chunks} | {'Ready' if is_processed else 'Processing...'}",
                ):
                    st.session_state.selected_paper = paper
                    st.session_state.messages = []
                    st.session_state.session_id = None
                    st.rerun()

            with col2:
                if st.button("🗑", key=f"del_{paper['id']}", help="Delete paper"):
                    delete_paper(paper["id"])
                    st.session_state.papers = list_papers()
                    if st.session_state.selected_paper and \
                       st.session_state.selected_paper["id"] == paper["id"]:
                        st.session_state.selected_paper = None
                        st.session_state.messages = []
                    st.rerun()

    st.divider()
    # st.markdown("*Built with FastAPI + FAISS + BGE + Groq*")


# --- Main Area ----------------------------------------------------------------

if not st.session_state.selected_paper:
    # Welcome screen
    st.markdown("# 🔬 Research RAG Assistant")
    st.markdown("### *Conversational AI for Research Papers*")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📄 Upload**\n\nUpload any research paper PDF from the sidebar")
    with col2:
        st.info("**💬 Ask**\n\nAsk questions in natural language about the paper")
    with col3:
        st.info("**📌 Cite**\n\nGet answers with section and page citations")

    st.divider()
    st.markdown("### How it works")
    st.markdown("""
    1. **Upload** a research PDF (IEEE, ACM, Springer, arXiv)
    2. System **extracts** text and splits into chunks
    3. **BGE embeddings** convert chunks to vectors stored in FAISS
    4. Your **question** is embedded and matched semantically
    5. **BGE Reranker** picks the top 3 most relevant chunks
    6. **Groq LLM** (Llama 3.3 70B) generates a grounded answer
    7. Citations with **page numbers** are returned
    """)

else:
    paper = st.session_state.selected_paper

    # Refresh paper status
    fresh = get_paper(paper["id"])
    if fresh:
        paper = fresh
        st.session_state.selected_paper = fresh

    # Paper Header
    title = paper.get("title") or paper.get("filename", "Unknown")
    st.markdown(f"## 📄 {title}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status = "✅ Ready" if paper.get("is_processed") else "⏳ Processing..."
        st.metric("Status", status)
    with col2:
        st.metric("Chunks", paper.get("chunk_count", 0))
    with col3:
        st.metric("Pages", paper.get("page_count") or "—")
    with col4:
        year = paper.get("year") or "—"
        st.metric("Year", year)

    if not paper.get("is_processed"):
        st.warning("⏳ Paper is still being indexed. Please wait ~30 seconds and click Refresh in the sidebar.")
        st.stop()

    st.divider()

    # Tabs
    tab_chat, tab_summary, tab_info = st.tabs(["💬 Chat", "📋 Summary", "ℹ️ Paper Info"])

    # ── Chat Tab ──────────────────────────────────────────────────────────────
    with tab_chat:

        # Suggested questions
        if not st.session_state.messages:
            st.markdown("**Suggested questions:**")
            suggestions = [
                "What is the main contribution of this paper?",
                "What dataset and evaluation metrics were used?",
                "Explain the proposed methodology.",
                "What are the limitations of this work?",
            ]
            cols = st.columns(2)
            for i, s in enumerate(suggestions):
                if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": s})
                    with st.spinner("Thinking..."):
                        result, err = send_chat(
                            paper["id"], s, st.session_state.session_id
                        )
                        if err:
                            st.error(f"Error: {err}")
                        else:
                            st.session_state.session_id = result["session_id"]
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result["answer"],
                                "citations": result.get("citations", []),
                                "confidence_score": result.get("confidence_score", 0),
                            })
                    st.rerun()

        # Display chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🔬"):
                    st.write(msg["content"])

                    # Confidence score
                    score = msg.get("confidence_score", 0)
                    if score:
                        score_clamped = max(0.0, min(1.0, abs(score)))
                        st.progress(score_clamped, text=f"Confidence: {round(score_clamped * 100)}%")

                    # Citations
                    citations = msg.get("citations", [])
                    if citations:
                        with st.expander(f"📌 {len(citations)} Source Citation(s)"):
                            for i, c in enumerate(citations):
                                section = c.get("section", "Section")
                                page = c.get("page_number")
                                snippet = c.get("text_snippet", "")
                                score_c = c.get("relevance_score", 0)

                                st.markdown(f"""
<div class="citation-box">
<strong>[{i+1}] {section}{f' · Page {page}' if page else ''}</strong>
· Relevance: {round(abs(score_c) * 100)}%<br><br>
<em>"{snippet[:300]}..."</em>
</div>
""", unsafe_allow_html=True)

        # Chat input
        st.divider()
        user_input = st.chat_input("Ask anything about this paper...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("🔍 Retrieving and reasoning..."):
                result, err = send_chat(
                    paper["id"], user_input, st.session_state.session_id
                )
            if err:
                st.error(f"Error: {err}")
            else:
                st.session_state.session_id = result["session_id"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result.get("citations", []),
                    "confidence_score": result.get("confidence_score", 0),
                })
            st.rerun()

        # Clear chat button
        if st.session_state.messages:
            if st.button("🗑 Clear Chat", use_container_width=False):
                st.session_state.messages = []
                st.session_state.session_id = None
                st.rerun()

    # ── Summary Tab ───────────────────────────────────────────────────────────
    with tab_summary:
        if st.button("✨ Generate Summary", type="primary", use_container_width=False):
            with st.spinner("Generating summary... (this takes ~30 seconds)"):
                result, err = summarize_paper(paper["id"])
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.session_state[f"summary_{paper['id']}"] = result

        summary_key = f"summary_{paper['id']}"
        if summary_key in st.session_state:
            summary = st.session_state[summary_key]
            sections = summary.get("sections", {})
            for section_name, section_text in sections.items():
                with st.expander(f"📖 {section_name}", expanded=True):
                    st.write(section_text)
        else:
            st.info("Click 'Generate Summary' to get an AI-generated summary of this paper.")

    # ── Info Tab ──────────────────────────────────────────────────────────────
    with tab_info:
        st.markdown("### Paper Details")

        if paper.get("title"):
            st.markdown(f"**Title:** {paper['title']}")
        if paper.get("authors"):
            st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
        if paper.get("year"):
            st.markdown(f"**Year:** {paper['year']}")
        if paper.get("abstract"):
            st.markdown("**Abstract:**")
            st.info(paper["abstract"])

        st.markdown("### Technical Details")
        st.json({
            "paper_id": paper["id"],
            "filename": paper["filename"],
            "chunk_count": paper.get("chunk_count", 0),
            "page_count": paper.get("page_count"),
            "file_size_bytes": paper.get("file_size_bytes"),
            "is_processed": paper.get("is_processed"),
        })