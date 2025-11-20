"""
Smart Document Intelligence - Modern Chat Interface
Single-page app with chat at center, file upload integrated
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
    page_title="Smart Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern, clean design
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Modern chat container */
    .main-chat {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }

    /* Chat messages */
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.75rem;
        animation: fadeIn 0.3s;
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }

    .assistant-message {
        background: #f7f7f8;
        margin-right: 20%;
    }

    /* Document chips */
    .doc-chip {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background: #e8f4f8;
        border-radius: 1rem;
        font-size: 0.9rem;
        border: 1px solid #3498db;
    }

    /* Quick actions */
    .quick-action {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s;
    }

    .quick-action:hover {
        border-color: #667eea;
        transform: translateX(4px);
    }

    /* Sidebar styling */
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None


def render_sidebar():
    """Render modern sidebar"""
    with st.sidebar:
        st.markdown("# 📄 SmartDoc")
        st.caption("AI-Powered Document Intelligence")

        st.markdown("---")

        # Documents section
        st.markdown("### 📚 Documents")

        if st.session_state.documents:
            for doc in st.session_state.documents[-5:]:  # Show last 5
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.caption(f"📄 {doc['name'][:25]}...")
                    with col2:
                        if st.button("×", key=f"remove_{doc['id']}", help="Remove"):
                            st.session_state.documents.remove(doc)
                            st.rerun()
        else:
            st.info("No documents uploaded yet")

        st.markdown("---")

        # Quick actions
        st.markdown("### ⚡ Quick Actions")

        actions = [
            ("📝", "Summarize", "summarize"),
            ("🔍", "Extract Entities", "entities"),
            ("🔄", "Compare Docs", "compare"),
            ("📚", "Generate Citation", "citation"),
        ]

        for icon, label, action in actions:
            if st.button(f"{icon} {label}", key=f"action_{action}", use_container_width=True):
                handle_quick_action(action)

        st.markdown("---")

        # Settings
        st.markdown("### ⚙️ Settings")

        llm_mode = st.radio(
            "LLM Mode",
            ["Auto", "Local (Ollama)", "Cloud (Gemini)"],
            key="llm_mode",
            label_visibility="collapsed"
        )

        show_sources = st.checkbox("Show sources", value=True, key="show_sources")

        st.markdown("---")

        # Stats
        st.markdown("### 📊 Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            st.metric("Docs", len(st.session_state.documents))

        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_chat():
    """Render main chat interface"""
    st.markdown('<div class="main-chat">', unsafe_allow_html=True)

    # Welcome message
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">📄 Smart Document Intelligence</h1>
            <p style="font-size: 1.2rem; color: #666;">Upload documents and ask anything</p>
        </div>
        """, unsafe_allow_html=True)

        # Quick start suggestions
        st.markdown("### 💡 Try asking:")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Upload a document", use_container_width=True):
                st.info("Click the 📎 button below to upload files")
            if st.button("🔍 Extract key information", use_container_width=True):
                add_message("Extract all key information from the documents")

        with col2:
            if st.button("📝 Summarize documents", use_container_width=True):
                add_message("Summarize the main points")
            if st.button("❓ Ask a question", use_container_width=True):
                st.info("Type your question in the chat box below")

    # Display messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
                # Show attached files
                if "files" in msg:
                    for file in msg["files"]:
                        st.markdown(f'<span class="doc-chip">📎 {file}</span>', unsafe_allow_html=True)

        else:  # assistant
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

                # Show sources if available
                if st.session_state.show_sources and "sources" in msg and msg["sources"]:
                    with st.expander(f"📚 {len(msg['sources'])} sources"):
                        for i, source in enumerate(msg["sources"][:3], 1):
                            st.markdown(f"**{i}.** `{source['doc_id']}` (score: {source['score']:.2f})")
                            st.code(source['text'][:150] + "...")

                # Show action buttons if present
                if "actions" in msg:
                    cols = st.columns(len(msg["actions"]))
                    for col, (label, action) in zip(cols, msg["actions"].items()):
                        with col:
                            if st.button(label, key=f"action_{msg['id']}_{action}"):
                                handle_action(action)

    st.markdown('</div>', unsafe_allow_html=True)


def render_input():
    """Render chat input with file upload"""
    col1, col2 = st.columns([6, 1])

    with col1:
        user_input = st.chat_input("Ask anything about your documents...")

    with col2:
        # File upload button (styled)
        uploaded_files = st.file_uploader(
            "📎",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"file_upload_{len(st.session_state.messages)}"
        )

    # Handle file upload
    if uploaded_files:
        process_uploads(uploaded_files)

    # Handle text input
    if user_input:
        handle_user_input(user_input)


def process_uploads(uploaded_files):
    """Process uploaded files in background"""
    with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
        from backend.pipeline import DocumentPipeline

        try:
            pipeline = DocumentPipeline(load_ocr_model=False, enable_vectordb=True)

            processed_files = []

            for file in uploaded_files:
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name

                # Process based on type
                if file.name.endswith('.pdf'):
                    result = pipeline.process_pdf(tmp_path, prompt_type="document")
                else:
                    result = pipeline.process_image(tmp_path, prompt_type="document", enhance=True)

                # Add to documents
                st.session_state.documents.append({
                    'id': result['doc_id'],
                    'name': file.name,
                    'type': 'pdf' if file.name.endswith('.pdf') else 'image',
                    'uploaded': datetime.now().strftime("%H:%M")
                })

                processed_files.append(file.name)

                # Clean up
                os.unlink(tmp_path)

            # Add system message
            file_list = ", ".join(processed_files)
            add_message(
                f"Uploaded and processed: {file_list}",
                role="assistant",
                actions={
                    "📝 Summarize": "summarize",
                    "🔍 Extract Info": "entities",
                    "❓ Ask Question": "question"
                }
            )

            st.rerun()

        except Exception as e:
            st.error(f"Error processing files: {e}")


def handle_user_input(user_input):
    """Handle user chat input"""
    # Add user message
    add_message(user_input, role="user")

    # Check if there are documents
    if not st.session_state.documents:
        add_message(
            "Please upload some documents first! Click the 📎 button to attach files.",
            role="assistant"
        )
        st.rerun()
        return

    # Process with RAG
    with st.spinner("Thinking..."):
        try:
            from backend.features.rag_pipeline import CompleteRAGPipeline
            from backend.llm.query_router import LLMType

            # Initialize RAG if needed
            if st.session_state.rag_pipeline is None:
                st.session_state.rag_pipeline = CompleteRAGPipeline(
                    collection_name="documents",
                    prefer_local=(st.session_state.llm_mode == "Local (Ollama)")
                )

            # Map LLM mode
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

            # Add assistant message
            add_message(
                response.answer,
                role="assistant",
                sources=response.sources if st.session_state.show_sources else None
            )

            st.rerun()

        except Exception as e:
            add_message(
                f"Sorry, I encountered an error: {str(e)}",
                role="assistant"
            )
            st.rerun()


def add_message(content, role="user", **kwargs):
    """Add message to chat history"""
    message = {
        "id": len(st.session_state.messages),
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M"),
        **kwargs
    }
    st.session_state.messages.append(message)


def handle_quick_action(action):
    """Handle quick action buttons"""
    if not st.session_state.documents:
        st.warning("Upload documents first!")
        return

    actions = {
        "summarize": "Summarize all documents in bullet points",
        "entities": "Extract all entities (people, organizations, dates, emails) from the documents",
        "compare": "Compare the documents and highlight similarities and differences",
        "citation": "Generate APA citations for all documents"
    }

    if action in actions:
        add_message(actions[action], role="user")
        st.rerun()


def handle_action(action):
    """Handle inline action buttons"""
    # Implement specific actions
    pass


def main():
    """Main application"""
    init_session_state()
    render_sidebar()
    render_chat()
    render_input()


if __name__ == "__main__":
    main()
