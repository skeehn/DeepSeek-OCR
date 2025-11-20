"""
Chat & Q&A Page
Interactive chat interface with RAG
"""
import streamlit as st
from pathlib import Path
import sys

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))


def show_chat_page():
    """Display chat page"""
    st.title("💬 Chat & Q&A")
    st.markdown("Ask questions about your documents and get AI-powered answers")

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize RAG pipeline
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None

    # Sidebar options
    with st.sidebar:
        st.markdown("### 🎛️ Chat Options")

        # Collection selection
        try:
            from backend.vectordb.chroma_manager import ChromaManager

            chroma = ChromaManager()
            collections = chroma.list_collections()

            if collections:
                collection = st.selectbox(
                    "Document Collection",
                    collections,
                    help="Select which document collection to search"
                )
            else:
                st.warning("No collections found. Upload and process documents first.")
                collection = "documents"

        except Exception as e:
            st.error(f"Error loading collections: {e}")
            collection = "documents"

        # LLM selection
        llm_type = st.radio(
            "LLM Selection",
            ["Auto", "Local (Ollama)", "Cloud (Gemini)"],
            help="Auto: Intelligent routing\nLocal: Privacy-focused\nCloud: Better for complex queries"
        )

        # Query settings
        top_k = st.slider(
            "Number of sources",
            min_value=1,
            max_value=10,
            value=5,
            help="How many document chunks to retrieve"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )

        include_sources = st.checkbox(
            "Show sources",
            value=True,
            help="Display source documents and relevance scores"
        )

        st.markdown("---")

        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Main chat interface
    col1, col2 = st.columns([3, 1])

    with col1:
        # Chat container
        chat_container = st.container()

        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(message["content"])

                        # Show sources if available
                        if include_sources and "sources" in message:
                            with st.expander("📚 Sources"):
                                for i, source in enumerate(message["sources"][:3], 1):
                                    st.markdown(f"**Source {i}** (Relevance: {source['score']:.2f})")
                                    st.markdown(f"Doc: `{source['doc_id']}`")
                                    st.code(source['text'][:200] + "...")
                                    st.markdown("---")

        # Chat input
        user_input = st.chat_input("Ask a question about your documents...")

        if user_input:
            handle_user_input(
                user_input,
                collection,
                llm_type,
                top_k,
                temperature,
                include_sources
            )

    with col2:
        # Quick questions
        st.markdown("### 💡 Suggested Questions")

        quick_questions = [
            "Summarize the main points",
            "What are the key findings?",
            "List all important dates",
            "Extract contact information",
            "What is the conclusion?",
            "Compare the approaches mentioned"
        ]

        for question in quick_questions:
            if st.button(question, use_container_width=True, key=f"quick_{question}"):
                handle_user_input(
                    question,
                    collection,
                    llm_type,
                    top_k,
                    temperature,
                    include_sources
                )

        st.markdown("---")

        # Stats
        st.markdown("### 📊 Chat Statistics")
        st.metric("Messages", len(st.session_state.chat_history))
        user_msgs = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        st.metric("Questions Asked", user_msgs)


def handle_user_input(user_input, collection, llm_type, top_k, temperature, include_sources):
    """Handle user input and generate response"""
    from backend.features.rag_pipeline import CompleteRAGPipeline
    from backend.llm.query_router import LLMType

    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    # Show thinking indicator
    with st.spinner("🤔 Thinking..."):
        try:
            # Initialize RAG pipeline if needed
            if st.session_state.rag_pipeline is None:
                st.session_state.rag_pipeline = CompleteRAGPipeline(
                    collection_name=collection,
                    prefer_local=(llm_type == "Local (Ollama)")
                )

            # Map LLM selection
            llm_type_map = {
                "Auto": LLMType.AUTO,
                "Local (Ollama)": LLMType.OLLAMA,
                "Cloud (Gemini)": LLMType.GEMINI
            }

            # Query
            response = st.session_state.rag_pipeline.query(
                query=user_input,
                top_k=top_k,
                llm_type=llm_type_map[llm_type],
                temperature=temperature,
                include_sources=include_sources
            )

            # Add assistant response to history
            assistant_message = {
                "role": "assistant",
                "content": response.answer
            }

            if include_sources and response.sources:
                assistant_message["sources"] = response.sources

            st.session_state.chat_history.append(assistant_message)

            # Rerun to update UI
            st.rerun()

        except Exception as e:
            st.error(f"Error generating response: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


def show_conversational_mode():
    """Show conversational RAG mode"""
    st.markdown("### 🔄 Conversational Mode")

    st.info("""
    **Conversational mode** maintains context across multiple questions.

    The system remembers your previous questions and answers, allowing for follow-up questions like:
    - "Tell me more about that"
    - "What else did it say?"
    - "Can you elaborate?"
    """)

    if st.button("Enable Conversational Mode"):
        st.session_state.conversational_mode = True
        st.success("Conversational mode enabled!")


if __name__ == "__main__":
    show_chat_page()
