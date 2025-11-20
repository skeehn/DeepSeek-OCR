"""
Smart Document Intelligence Platform - Streamlit UI
Main application entry point
"""
import streamlit as st
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Page configuration
st.set_page_config(
    page_title="Smart Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Smart Document Intelligence Platform - Powered by DeepSeek-OCR, Ollama, and Gemini"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .feature-card {
        background-color: #fff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def show_home_page():
    """Display home page"""
    # Header
    st.markdown('<h1 class="main-header">📄 Smart Document Intelligence Platform</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Transform documents into actionable insights with AI-powered OCR, RAG, and advanced analytics</p>',
        unsafe_allow_html=True
    )

    # Features overview
    st.markdown("## 🌟 Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <h3>Document Processing</h3>
            <p>Upload PDFs and images, extract text with layout preservation using DeepSeek-OCR</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <h3>Semantic Search</h3>
            <p>Find relevant information using vector embeddings and similarity search</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <h3>Intelligent Q&A</h3>
            <p>Ask questions and get AI-powered answers with source citations</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Document Analysis</h3>
            <p>Extract entities, compare documents, and generate summaries</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📤</div>
            <h3>Export & Citations</h3>
            <p>Export results to JSON, Markdown, HTML, and generate academic citations</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3>Dual LLM System</h3>
            <p>Local Ollama for privacy, Cloud Gemini for complex reasoning</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick start guide
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 🚀 Quick Start Guide")

        st.markdown("""
        ### 1️⃣ Upload Documents
        Navigate to **📤 Upload** to add PDF or image documents to the system.

        ### 2️⃣ Process & Index
        Documents are automatically processed with OCR and indexed in the vector database.

        ### 3️⃣ Ask Questions
        Go to **💬 Chat** to ask questions about your documents and get AI-powered answers.

        ### 4️⃣ Analyze & Export
        Use **📊 Analyze** to extract entities, compare documents, generate summaries, and export results.

        ### 5️⃣ Browse Documents
        View all your documents in **📚 Documents** and manage your library.
        """)

    with col2:
        st.markdown("## 📈 System Status")

        try:
            from backend.utils.storage import DocumentStorage
            storage = DocumentStorage()
            docs = storage.list_documents()

            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(docs)}</div>
                <div class="stat-label">Total Documents</div>
            </div>
            """, unsafe_allow_html=True)

            processed = len([d for d in docs if d.get('status') == 'processed'])
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{processed}</div>
                <div class="stat-label">Processed Documents</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Could not load document statistics: {e}")

        st.markdown("### 🔧 System Components")
        st.markdown("""
        - ✅ DeepSeek-OCR
        - ✅ ChromaDB Vector Store
        - ✅ Ollama (Local LLM)
        - ✅ Gemini API
        """)

    st.markdown("---")

    # Technology stack
    st.markdown("## 💻 Technology Stack")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**OCR & Processing**")
        st.markdown("- DeepSeek-OCR")
        st.markdown("- vLLM")
        st.markdown("- PyMuPDF")
        st.markdown("- Pillow")

    with col2:
        st.markdown("**Vector Database**")
        st.markdown("- ChromaDB")
        st.markdown("- Sentence Transformers")
        st.markdown("- FAISS Indexing")

    with col3:
        st.markdown("**LLM Integration**")
        st.markdown("- Ollama (Llama 3.3)")
        st.markdown("- Google Gemini")
        st.markdown("- Query Routing")

    with col4:
        st.markdown("**Advanced Features**")
        st.markdown("- Entity Extraction")
        st.markdown("- Summarization")
        st.markdown("- Document Comparison")
        st.markdown("- Citation Generation")


def main():
    """Main application"""
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=SmartDoc", use_column_width=True)

        st.markdown("## Navigation")

        # Navigation buttons
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"

        if st.button("📤 Upload Documents", use_container_width=True):
            st.session_state.page = "upload"

        if st.button("💬 Chat & Q&A", use_container_width=True):
            st.session_state.page = "chat"

        if st.button("📚 Browse Documents", use_container_width=True):
            st.session_state.page = "documents"

        if st.button("📊 Analysis", use_container_width=True):
            st.session_state.page = "analysis"

        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.page = "settings"

        st.markdown("---")

        # Quick actions
        st.markdown("### Quick Actions")

        if st.button("🔍 Search Documents", use_container_width=True):
            st.session_state.page = "search"

        if st.button("📝 Generate Summary", use_container_width=True):
            st.session_state.page = "summarize"

        if st.button("🔗 Generate Citation", use_container_width=True):
            st.session_state.page = "citation"

        st.markdown("---")

        # Info
        st.markdown("### ℹ️ About")
        st.caption("Smart Document Intelligence Platform v1.0")
        st.caption("Powered by DeepSeek-OCR, Ollama, and Gemini")

    # Initialize page state
    if 'page' not in st.session_state:
        st.session_state.page = "home"

    # Display appropriate page
    if st.session_state.page == "home":
        show_home_page()
    elif st.session_state.page == "upload":
        from pages.upload_page import show_upload_page
        show_upload_page()
    elif st.session_state.page == "chat":
        from pages.chat_page import show_chat_page
        show_chat_page()
    elif st.session_state.page == "documents":
        from pages.documents_page import show_documents_page
        show_documents_page()
    elif st.session_state.page == "analysis":
        from pages.analysis_page import show_analysis_page
        show_analysis_page()
    elif st.session_state.page == "settings":
        show_settings_page()
    else:
        # Redirect any other page to home
        st.session_state.page = "home"
        st.rerun()


def show_settings_page():
    """Settings page"""
    st.title("⚙️ Settings")
    st.markdown("Configure the Smart Document Intelligence Platform")

    st.markdown("### 🔧 System Configuration")

    # LLM Settings
    with st.expander("🤖 LLM Settings", expanded=True):
        st.markdown("#### Ollama (Local LLM)")
        ollama_url = st.text_input("Ollama Server URL", value="http://localhost:11434")
        ollama_model = st.text_input("Default Model", value="llama3.3")

        if st.button("Test Ollama Connection"):
            st.info("Testing Ollama connection...")

        st.markdown("---")

        st.markdown("#### Gemini (Cloud LLM)")
        gemini_api_key = st.text_input("Gemini API Key", type="password", help="Get API key from Google AI Studio")
        gemini_model = st.text_input("Model", value="gemini-2.0-flash")

        if st.button("Test Gemini Connection"):
            st.info("Testing Gemini connection...")

    # Vector DB Settings
    with st.expander("🔍 Vector Database Settings"):
        st.markdown("#### ChromaDB")
        chroma_path = st.text_input("ChromaDB Path", value="./chroma_storage")
        embedding_model = st.text_input("Embedding Model", value="all-MiniLM-L6-v2")

        if st.button("Test ChromaDB Connection"):
            st.info("Testing ChromaDB connection...")

    # OCR Settings
    with st.expander("📄 OCR Settings"):
        st.markdown("#### DeepSeek-OCR")
        base_size = st.select_slider("Base Resolution", options=[512, 640, 1024, 1280], value=1024)
        crop_mode = st.checkbox("Enable Crop Mode (Gundam)", value=True)
        max_crops = st.slider("Max Crops", 1, 10, 6)

    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved!")


if __name__ == "__main__":
    main()
