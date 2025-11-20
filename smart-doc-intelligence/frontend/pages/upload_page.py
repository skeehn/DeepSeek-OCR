"""
Document Upload Page
Handles document upload and processing
"""
import streamlit as st
from pathlib import Path
import sys
import tempfile
import os

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))


def show_upload_page():
    """Display upload page"""
    st.title("📤 Upload Documents")
    st.markdown("Upload PDF or image files to process with DeepSeek-OCR")

    # Upload options
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Upload Files")

        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF or image files",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Supported formats: PDF, PNG, JPG, JPEG"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")

            # Show file details
            with st.expander("📋 File Details"):
                for i, file in enumerate(uploaded_files, 1):
                    file_size = len(file.getvalue()) / 1024  # KB
                    st.write(f"{i}. **{file.name}** ({file_size:.1f} KB)")

    with col2:
        st.markdown("### Processing Options")

        # OCR options
        prompt_type = st.selectbox(
            "Prompt Type",
            ["document", "free", "figure", "detail"],
            help="document: Convert to markdown\nfree: Free OCR\nfigure: Parse figures\ndetail: Detailed description"
        )

        enhance_images = st.checkbox(
            "Enhance Images",
            value=True,
            help="Apply auto-contrast and sharpening"
        )

        enable_vectordb = st.checkbox(
            "Index in Vector Database",
            value=True,
            help="Enable semantic search and RAG"
        )

        chunk_strategy = st.selectbox(
            "Chunking Strategy",
            ["paragraph", "sentence", "fixed", "markdown"],
            help="How to split the text for indexing"
        )

    # Processing button
    if uploaded_files:
        st.markdown("---")

        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            process_documents(
                uploaded_files,
                prompt_type,
                enhance_images,
                enable_vectordb,
                chunk_strategy
            )

    # Recent uploads
    st.markdown("---")
    st.markdown("### 📋 Recent Uploads")

    try:
        from backend.utils.storage import DocumentStorage

        storage = DocumentStorage()
        docs = storage.list_documents()

        if docs:
            # Show last 5 documents
            recent_docs = sorted(docs, key=lambda x: x.get('upload_date', ''), reverse=True)[:5]

            for doc in recent_docs:
                with st.expander(f"📄 {doc.get('filename', 'Unknown')}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write(f"**Doc ID:** `{doc.get('doc_id', 'N/A')}`")
                        st.write(f"**Type:** {doc.get('doc_type', 'N/A')}")

                    with col2:
                        st.write(f"**Status:** {doc.get('status', 'N/A')}")
                        st.write(f"**Upload Date:** {doc.get('upload_date', 'N/A')}")

                    with col3:
                        if doc.get('page_count'):
                            st.write(f"**Pages:** {doc['page_count']}")
                        if doc.get('word_count'):
                            st.write(f"**Words:** {doc['word_count']}")

                    if st.button(f"View Document", key=f"view_{doc.get('doc_id')}"):
                        st.session_state.page = "documents"
                        st.session_state.selected_doc = doc.get('doc_id')
                        st.rerun()
        else:
            st.info("No documents uploaded yet")

    except Exception as e:
        st.error(f"Error loading recent uploads: {e}")


def process_documents(uploaded_files, prompt_type, enhance_images, enable_vectordb, chunk_strategy):
    """Process uploaded documents"""
    from backend.pipeline import DocumentPipeline
    from backend.utils.storage import DocumentStorage

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Initialize pipeline
        status_text.text("Initializing pipeline...")
        pipeline = DocumentPipeline(load_ocr_model=False, enable_vectordb=enable_vectordb)
        storage = DocumentStorage()

        total_files = len(uploaded_files)

        results = []

        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing {i+1}/{total_files}: {uploaded_file.name}")

            try:
                # Save uploaded file to temp directory
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Determine file type
                file_ext = Path(uploaded_file.name).suffix.lower()

                if file_ext == '.pdf':
                    # Process PDF
                    result = pipeline.process_pdf(
                        tmp_path,
                        prompt_type=prompt_type
                    )
                else:
                    # Process image
                    result = pipeline.process_image(
                        tmp_path,
                        prompt_type=prompt_type,
                        enhance=enhance_images
                    )

                results.append({
                    "filename": uploaded_file.name,
                    "doc_id": result.get("doc_id"),
                    "status": "success",
                    "word_count": result.get("word_count", 0),
                    "chunk_count": result.get("chunk_count", 0)
                })

                # Clean up temp file
                os.unlink(tmp_path)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
                results.append({
                    "filename": uploaded_file.name,
                    "status": "error",
                    "error": str(e)
                })

        # Show results
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")

        st.success(f"Successfully processed {len([r for r in results if r['status'] == 'success'])}/{total_files} documents")

        # Show results table
        with st.expander("📊 Processing Results"):
            for result in results:
                if result['status'] == 'success':
                    st.success(f"✅ **{result['filename']}**")
                    st.write(f"   - Doc ID: `{result['doc_id']}`")
                    st.write(f"   - Words: {result.get('word_count', 'N/A')}")
                    st.write(f"   - Chunks: {result.get('chunk_count', 'N/A')}")
                else:
                    st.error(f"❌ **{result['filename']}**: {result.get('error', 'Unknown error')}")

        # Navigation button
        if st.button("📚 Go to Documents"):
            st.session_state.page = "documents"
            st.rerun()

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    show_upload_page()
