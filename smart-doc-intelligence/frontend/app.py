"""
Smart Document Intelligence - Ultra-Modern Chat Interface
10x better: Clean design, smart features, perfect UX
"""
import streamlit as st
from pathlib import Path
import sys
import tempfile
import os
from datetime import datetime

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

    /* Chat container - centered, card style */
    .chat-container {
        max-width: 800px;
        margin: 2rem auto;
        background: white;
        border-radius: 24px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        overflow: hidden;
    }

    /* Header bar */
    .header-bar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Chat messages */
    .chat-messages {
        padding: 2rem;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }

    /* Message bubbles */
    .message {
        margin: 1rem 0;
        animation: slideIn 0.3s ease-out;
    }

    .user-msg {
        display: flex;
        justify-content: flex-end;
    }

    .user-msg .bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 4px 20px;
        max-width: 70%;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }

    .assistant-msg .bubble {
        background: #f7f8fa;
        color: #2d3748;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 4px;
        max-width: 70%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Document chips */
    .doc-chip {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.4rem 0.8rem;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 0.25rem;
        backdrop-filter: blur(10px);
    }

    /* Action buttons */
    .action-btn {
        display: inline-block;
        background: white;
        color: #667eea;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        margin: 0.25rem;
        font-size: 0.9rem;
        border: 1px solid #667eea;
        cursor: pointer;
        transition: all 0.2s;
    }

    .action-btn:hover {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
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

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .typing {
        animation: pulse 1.5s infinite;
    }

    /* Input area */
    .stChatInput {
        border-radius: 20px !important;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: white;
    }

    /* Quick actions grid */
    .quick-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin: 1rem 0;
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

    /* File upload zone */
    .upload-zone {
        border: 2px dashed #cbd5e0;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: #f7f8fa;
        transition: all 0.3s;
        cursor: pointer;
    }

    .upload-zone:hover {
        border-color: #667eea;
        background: #eef2ff;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .status-success {
        background: #d4edda;
        color: #155724;
    }

    .status-processing {
        background: #fff3cd;
        color: #856404;
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
        'active_doc': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header():
    """Render top header"""
    col1, col2, col3 = st.columns([2, 3, 2])

    with col1:
        st.markdown("### ✨ SmartDoc AI")

    with col2:
        if st.session_state.documents:
            # Document tabs
            doc_names = [f"📄 {doc['name'][:15]}..." for doc in st.session_state.documents[-3:]]
            selected = st.pills("Active Documents", doc_names, selection_mode="single")

    with col3:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("⚙️", help="Settings"):
                st.session_state.show_settings = not st.session_state.get('show_settings', False)
        with col_b:
            if st.button("📤", help="Export Chat"):
                export_chat()
        with col_c:
            if st.button("🗑️", help="Clear"):
                st.session_state.messages = []
                st.rerun()


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
        help="Upload PDFs or images to get started"
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
                        st.session_state.documents = [d for d in st.session_state.documents if d['id'] != doc['id']]
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
            if st.button(label, use_container_width=True):
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
            key=f"upload_{len(st.session_state.messages)}"
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
    """Process uploaded files"""
    with st.spinner(f"✨ Processing {len(files)} file(s)..."):
        try:
            from backend.pipeline import DocumentPipeline

            pipeline = DocumentPipeline(load_ocr_model=False, enable_vectordb=True)
            processed = []

            for file in files:
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
                os.unlink(tmp_path)

            # Success message
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

    # Process with RAG
    with st.spinner("✨ Thinking..."):
        try:
            from backend.features.rag_pipeline import CompleteRAGPipeline
            from backend.llm.query_router import LLMType

            # Init RAG
            if st.session_state.rag_pipeline is None:
                st.session_state.rag_pipeline = CompleteRAGPipeline(
                    collection_name="documents",
                    prefer_local=(st.session_state.llm_mode == "Local (Ollama)")
                )

            # Map LLM
            llm_map = {
                "Auto": LLMType.AUTO,
                "Local (Ollama)": LLMType.OLLAMA,
                "Cloud (Gemini)": LLMType.GEMINI
            }

            # Query
            response = st.session_state.rag_pipeline.query(
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
            add_msg(f"❌ Error: {str(e)}", "assistant")
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
    actions = {
        "summarize": "Summarize all documents in bullet points",
        "entities": "Extract all entities (people, organizations, dates, emails)",
        "question": ""  # Let user ask
    }

    if action in actions and actions[action]:
        add_msg(actions[action], "user")
        st.rerun()


def export_chat():
    """Export chat history"""
    if not st.session_state.messages:
        st.warning("No messages to export")
        return

    # Create export
    export_text = "# SmartDoc AI Chat Export\n\n"
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "AI"
        export_text += f"**{role}** ({msg['time']}):\n{msg['content']}\n\n"

    st.download_button(
        "📥 Download Chat",
        export_text,
        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )


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
