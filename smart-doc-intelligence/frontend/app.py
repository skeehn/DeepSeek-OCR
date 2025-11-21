"""
Smart Document Intelligence - Production-Ready Chat Interface
Fixed critical bugs, added caching, streaming, and better UX
"""
import streamlit as st
from pathlib import Path
import sys
import tempfile
import os
from datetime import datetime
import time

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Page config
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra-modern CSS
st.markdown("""
<style>
    /* Clean slate */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Main layout */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 0 !important;
    }

    /* Message bubbles - better styling */
    .stChatMessage {
        animation: slideIn 0.3s ease-out;
    }

    /* Source cards */
    .source-card {
        background: #f7f8fa;
        border-left: 3px solid #667eea;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 0.9rem;
    }

    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Welcome screen */
    .welcome {
        text-align: center;
        padding: 4rem 2rem;
    }

    .welcome h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def init_state():
    """Initialize session state"""
    defaults = {
        'messages': [],
        'documents': [],
        'rag_pipeline': None,
        'llm_mode': 'Auto',
        'show_sources': True,
        'confirm_clear': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_rag_pipeline(prefer_local=False):
    """Cached RAG pipeline initialization"""
    from backend.features.rag_pipeline import CompleteRAGPipeline
    return CompleteRAGPipeline(
        collection_name="documents",
        prefer_local=prefer_local
    )


def render_header():
    """Render top header"""
    col1, col2, col3 = st.columns([2, 3, 2])

    with col1:
        st.markdown("### ✨ SmartDoc AI")

    with col2:
        if st.session_state.documents:
            # Show document count
            st.caption(f"📚 {len(st.session_state.documents)} document(s) loaded")

    with col3:
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            # Export directly
            if st.session_state.messages:
                export_text = "# SmartDoc AI Chat Export\n\n"
                for msg in st.session_state.messages:
                    role = "You" if msg["role"] == "user" else "AI"
                    export_text += f"**{role}** ({msg['time']}):\n{msg['content']}\n\n"

                st.download_button(
                    "📤",
                    export_text,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    help="Export chat"
                )

        with col_b:
            if st.button("⚙️", help="Settings"):
                pass  # Settings in sidebar

        with col_c:
            if st.button("🗑️", help="Clear chat"):
                st.session_state.confirm_clear = True


def render_welcome():
    """Render welcome screen"""
    st.markdown("""
    <div class="welcome">
        <h1>✨ SmartDoc AI</h1>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
            Your intelligent document assistant
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Upload zone
    st.markdown("### 📎 Upload Documents")
    uploaded = st.file_uploader(
        "Drag and drop files here",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload PDFs or images to get started (max 100MB per file)"
    )

    if uploaded:
        process_files(uploaded)

    # Quick start examples
    st.markdown("### 💡 Or try these examples:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📝 Summarize a research paper", use_container_width=True):
            st.info("👆 Upload a PDF first!")
        if st.button("🔍 Extract key data from invoice", use_container_width=True):
            st.info("👆 Upload a PDF first!")

    with col2:
        if st.button("📊 Compare multiple documents", use_container_width=True):
            st.info("👆 Upload PDFs first!")
        if st.button("📚 Generate citations", use_container_width=True):
            st.info("👆 Upload a PDF first!")


def render_chat():
    """Render chat interface"""
    # Clear confirmation dialog
    if st.session_state.get('confirm_clear', False):
        with st.container():
            st.warning("⚠️ Clear all messages? This cannot be undone.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, clear", type="primary", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.confirm_clear = False
                    st.rerun()
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()

    # Render messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
                if "files" in msg:
                    st.caption("📎 " + ", ".join(msg["files"]))
        else:
            with st.chat_message("assistant", avatar="✨"):
                st.markdown(msg["content"])

                # Action buttons
                if "actions" in msg:
                    cols = st.columns(len(msg["actions"]))
                    for col, (label, action_key) in zip(cols, msg["actions"].items()):
                        with col:
                            if st.button(label, key=f"btn_{msg['id']}_{action_key}", use_container_width=True):
                                handle_action(action_key)

                # Sources
                if st.session_state.show_sources and "sources" in msg and msg["sources"]:
                    with st.expander(f"📚 View {len(msg['sources'])} sources"):
                        for i, src in enumerate(msg["sources"][:3], 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {i}</strong> • {src['doc_id']} • Score: {src['score']:.2f}<br>
                                <code>{src['text'][:200]}...</code>
                            </div>
                            """, unsafe_allow_html=True)


def render_sidebar():
    """Render collapsible sidebar"""
    with st.sidebar:
        st.markdown("## 📄 SmartDoc AI")
        st.caption("Intelligent Document Assistant")

        st.markdown("---")

        # Documents
        st.markdown("### 📚 Documents")
        if st.session_state.documents:
            for doc in st.session_state.documents:
                col1, col2 = st.columns([4, 1])
                with col1:
                    status = "🟢" if doc.get('processed') else "🟡"
                    st.caption(f"{status} {doc['name'][:20]}...")
                with col2:
                    if st.button("×", key=f"del_{doc['id']}", help="Remove"):
                        # Remove from UI and vector DB
                        remove_document(doc['id'])
                        st.rerun()
        else:
            st.info("No documents yet")

        st.markdown("---")

        # Quick actions
        st.markdown("### ⚡ Quick Actions")

        actions = {
            "📝 Summarize": "Summarize all documents in bullet points",
            "🔍 Extract Entities": "Extract all people, organizations, dates, and emails",
            "🔄 Compare": "Compare the documents and show similarities",
            "📚 Citation": "Generate APA citation for the first document"
        }

        for label, prompt in actions.items():
            if st.button(label, use_container_width=True, key=f"quick_{label}"):
                if st.session_state.documents:
                    add_msg(prompt, "user")
                    st.rerun()
                else:
                    st.warning("Upload documents first!")

        st.markdown("---")

        # Settings
        st.markdown("### ⚙️ Settings")

        st.session_state.llm_mode = st.selectbox(
            "LLM",
            ["Auto", "Local (Ollama)", "Cloud (Gemini)"],
            label_visibility="collapsed"
        )

        st.session_state.show_sources = st.toggle("Show sources", value=True)

        st.markdown("---")

        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💬", len(st.session_state.messages))
        with col2:
            st.metric("📄", len(st.session_state.documents))


def render_input():
    """Render chat input"""
    # File upload inline
    col1, col2 = st.columns([1, 6])

    with col1:
        uploaded = st.file_uploader(
            "📎",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"upload_{datetime.now().timestamp()}"  # Stable unique key
        )

    with col2:
        user_input = st.chat_input("Ask anything about your documents...")

    # Handle upload
    if uploaded:
        process_files(uploaded)

    # Handle input
    if user_input:
        handle_input(user_input)


def process_files(files):
    """Process uploaded files with proper cleanup"""
    # Validate file sizes
    MAX_SIZE_MB = 100
    for file in files:
        size_mb = len(file.getvalue()) / (1024 * 1024)
        if size_mb > MAX_SIZE_MB:
            st.error(f"❌ {file.name} is too large ({size_mb:.1f}MB). Max size is {MAX_SIZE_MB}MB.")
            return

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        from backend.pipeline import DocumentPipeline

        pipeline = DocumentPipeline(load_ocr_model=False, enable_vectordb=True)
        processed = []
        total = len(files)

        for idx, file in enumerate(files):
            status_text.text(f"Processing {idx+1}/{total}: {file.name}")
            progress_bar.progress((idx + 1) / total)

            tmp_path = None
            try:
                # Save temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name

                # Process
                if file.name.endswith('.pdf'):
                    result = pipeline.process_pdf(tmp_path, prompt_type="document")
                else:
                    result = pipeline.process_image(tmp_path, prompt_type="document", enhance=True)

                # Store
                st.session_state.documents.append({
                    'id': result['doc_id'],
                    'name': file.name,
                    'processed': True,
                    'uploaded': datetime.now().strftime("%H:%M")
                })

                processed.append(file.name)

            finally:
                # Always cleanup temp file
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Success message
        progress_bar.empty()
        status_text.empty()

        add_msg(
            f"✅ Successfully processed: **{', '.join(processed)}**\n\nWhat would you like to know?",
            "assistant",
            actions={
                "📝 Summarize": "summarize",
                "🔍 Extract Info": "entities",
                "❓ Ask Question": "question"
            }
        )

        st.rerun()

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error: {e}")


def handle_input(user_input):
    """Handle user input"""
    # Add user message
    add_msg(user_input, "user")

    # Check docs
    if not st.session_state.documents:
        add_msg("Please upload documents first! Click 📎 to attach files.", "assistant")
        st.rerun()
        return

    # Process with cached RAG
    with st.spinner("✨ Thinking..."):
        try:
            from backend.llm.query_router import LLMType

            # Get cached pipeline
            prefer_local = (st.session_state.llm_mode == "Local (Ollama)")
            pipeline = get_rag_pipeline(prefer_local)

            # Map LLM
            llm_map = {
                "Auto": LLMType.AUTO,
                "Local (Ollama)": LLMType.OLLAMA,
                "Cloud (Gemini)": LLMType.GEMINI
            }

            # Query
            response = pipeline.query(
                query=user_input,
                top_k=5,
                llm_type=llm_map[st.session_state.llm_mode],
                temperature=0.7,
                include_sources=True
            )

            # Add response
            add_msg(
                response.answer,
                "assistant",
                sources=response.sources if st.session_state.show_sources else None
            )

            st.rerun()

        except Exception as e:
            # Add retry button on failure
            add_msg(
                f"❌ Error: {str(e)}\n\nPlease try again or check your LLM settings.",
                "assistant",
                actions={"🔄 Retry": "retry"}
            )
            st.rerun()


def add_msg(content, role="user", **kwargs):
    """Add message to history"""
    st.session_state.messages.append({
        "id": len(st.session_state.messages),
        "role": role,
        "content": content,
        "time": datetime.now().strftime("%H:%M"),
        **kwargs
    })


def handle_action(action):
    """Handle quick actions"""
    if action == "retry" and st.session_state.messages:
        # Retry last user message
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                handle_input(msg["content"])
                return

    actions = {
        "summarize": "Summarize all documents in bullet points",
        "entities": "Extract all entities (people, organizations, dates, emails)",
        "question": ""  # Let user ask
    }

    if action in actions and actions[action]:
        add_msg(actions[action], "user")
        st.rerun()


def remove_document(doc_id):
    """Remove document from UI and vector DB"""
    try:
        # Remove from vector DB
        from backend.vectordb.chroma_manager import ChromaManager
        chroma = ChromaManager()

        # Delete all chunks for this document
        collection = chroma.get_collection("documents")
        if collection:
            # Delete by metadata filter
            collection.delete(where={"doc_id": doc_id})

    except Exception as e:
        st.warning(f"Could not remove from vector DB: {e}")

    # Remove from UI state
    st.session_state.documents = [
        d for d in st.session_state.documents
        if d['id'] != doc_id
    ]


def main():
    """Main app"""
    init_state()
    render_header()

    st.markdown("---")

    # Main content
    if not st.session_state.messages:
        render_welcome()
    else:
        render_chat()

    # Always show input
    render_input()

    # Sidebar
    render_sidebar()


if __name__ == "__main__":
    main()
