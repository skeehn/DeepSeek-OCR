"""
Documents Browser Page
View and manage uploaded documents
"""
import streamlit as st
from pathlib import Path
import sys
import json

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))


def show_documents_page():
    """Display documents page"""
    st.title("📚 Document Library")
    st.markdown("Browse and manage your uploaded documents")

    # Load documents
    try:
        from backend.utils.storage import DocumentStorage

        storage = DocumentStorage()
        docs = storage.list_documents()

        if not docs:
            st.info("📭 No documents yet. Upload some documents to get started!")
            if st.button("📤 Go to Upload"):
                st.session_state.page = "upload"
                st.rerun()
            return

        # Sidebar filters
        with st.sidebar:
            st.markdown("### 🔍 Filters")

            # Status filter
            status_options = ["All"] + list(set(d.get('status', 'unknown') for d in docs))
            status_filter = st.selectbox("Status", status_options)

            # Type filter
            type_options = ["All"] + list(set(d.get('doc_type', 'unknown') for d in docs))
            type_filter = st.selectbox("Type", type_options)

            # Search
            search_query = st.text_input("🔍 Search by filename")

            st.markdown("---")

            # Sort options
            sort_by = st.selectbox(
                "Sort by",
                ["Upload Date (Newest)", "Upload Date (Oldest)", "Filename (A-Z)", "Filename (Z-A)"]
            )

        # Apply filters
        filtered_docs = docs

        if status_filter != "All":
            filtered_docs = [d for d in filtered_docs if d.get('status') == status_filter]

        if type_filter != "All":
            filtered_docs = [d for d in filtered_docs if d.get('doc_type') == type_filter]

        if search_query:
            filtered_docs = [
                d for d in filtered_docs
                if search_query.lower() in d.get('filename', '').lower()
            ]

        # Sort documents
        if sort_by == "Upload Date (Newest)":
            filtered_docs = sorted(filtered_docs, key=lambda x: x.get('upload_date', ''), reverse=True)
        elif sort_by == "Upload Date (Oldest)":
            filtered_docs = sorted(filtered_docs, key=lambda x: x.get('upload_date', ''))
        elif sort_by == "Filename (A-Z)":
            filtered_docs = sorted(filtered_docs, key=lambda x: x.get('filename', ''))
        elif sort_by == "Filename (Z-A)":
            filtered_docs = sorted(filtered_docs, key=lambda x: x.get('filename', ''), reverse=True)

        # Show statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Documents", len(docs))

        with col2:
            processed = len([d for d in docs if d.get('status') == 'processed'])
            st.metric("Processed", processed)

        with col3:
            total_pages = sum(d.get('page_count', 0) for d in docs)
            st.metric("Total Pages", total_pages)

        with col4:
            st.metric("Filtered Results", len(filtered_docs))

        st.markdown("---")

        # Display documents
        if filtered_docs:
            st.markdown(f"### Documents ({len(filtered_docs)})")

            # View mode
            view_mode = st.radio(
                "View Mode",
                ["Cards", "List", "Table"],
                horizontal=True
            )

            if view_mode == "Cards":
                show_card_view(filtered_docs, storage)
            elif view_mode == "List":
                show_list_view(filtered_docs, storage)
            else:
                show_table_view(filtered_docs, storage)

        else:
            st.warning("No documents match the filters")

    except Exception as e:
        st.error(f"Error loading documents: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())


def show_card_view(docs, storage):
    """Display documents as cards"""
    # Create columns for cards
    cols_per_row = 3
    rows = [docs[i:i+cols_per_row] for i in range(0, len(docs), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)

        for col, doc in zip(cols, row):
            with col:
                doc_id = doc.get('doc_id', 'unknown')
                filename = doc.get('filename', 'Unknown')
                status = doc.get('status', 'unknown')
                doc_type = doc.get('doc_type', 'unknown')

                # Status badge
                status_color = {
                    'uploaded': '🟡',
                    'processing': '🔵',
                    'processed': '🟢',
                    'error': '🔴'
                }.get(status, '⚪')

                with st.container():
                    st.markdown(f"""
                    <div style="
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 1rem;
                        margin-bottom: 1rem;
                        background-color: #f9f9f9;
                    ">
                        <h4 style="margin:0; font-size: 0.9rem;">{status_color} {filename[:30]}...</h4>
                        <p style="color: #666; font-size: 0.8rem; margin: 0.5rem 0;">
                            Type: {doc_type} | Status: {status}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("📄 View", key=f"view_card_{doc_id}", use_container_width=True):
                        show_document_detail(doc, storage)

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💬 Chat", key=f"chat_card_{doc_id}", use_container_width=True):
                            st.session_state.page = "chat"
                            st.session_state.selected_doc = doc_id
                            st.rerun()
                    with col2:
                        if st.button("📊 Analyze", key=f"analyze_card_{doc_id}", use_container_width=True):
                            st.session_state.page = "analysis"
                            st.session_state.selected_doc = doc_id
                            st.rerun()


def show_list_view(docs, storage):
    """Display documents as a list"""
    for doc in docs:
        doc_id = doc.get('doc_id', 'unknown')
        filename = doc.get('filename', 'Unknown')
        status = doc.get('status', 'unknown')
        upload_date = doc.get('upload_date', 'N/A')

        with st.expander(f"📄 {filename}"):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.write(f"**Doc ID:** `{doc_id}`")
                st.write(f"**Status:** {status}")
                st.write(f"**Upload Date:** {upload_date}")

            with col2:
                if doc.get('page_count'):
                    st.write(f"**Pages:** {doc['page_count']}")
                if doc.get('word_count'):
                    st.write(f"**Words:** {doc['word_count']}")
                if doc.get('chunk_count'):
                    st.write(f"**Chunks:** {doc['chunk_count']}")

            with col3:
                if st.button("📄 View Details", key=f"view_list_{doc_id}"):
                    show_document_detail(doc, storage)

                if st.button("💬 Chat About This", key=f"chat_list_{doc_id}"):
                    st.session_state.page = "chat"
                    st.session_state.selected_doc = doc_id
                    st.rerun()


def show_table_view(docs, storage):
    """Display documents as a table"""
    import pandas as pd

    # Prepare data for table
    table_data = []
    for doc in docs:
        table_data.append({
            "Filename": doc.get('filename', 'Unknown'),
            "Status": doc.get('status', 'unknown'),
            "Type": doc.get('doc_type', 'unknown'),
            "Pages": doc.get('page_count', 0),
            "Words": doc.get('word_count', 0),
            "Upload Date": doc.get('upload_date', 'N/A'),
            "Doc ID": doc.get('doc_id', 'unknown')
        })

    df = pd.DataFrame(table_data)

    # Display table with selection
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # Selection
    selected_doc_id = st.selectbox(
        "Select document to view",
        options=[d.get('doc_id') for d in docs],
        format_func=lambda x: next((d.get('filename') for d in docs if d.get('doc_id') == x), x)
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 View Details", use_container_width=True):
            selected_doc = next((d for d in docs if d.get('doc_id') == selected_doc_id), None)
            if selected_doc:
                show_document_detail(selected_doc, storage)

    with col2:
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.page = "chat"
            st.session_state.selected_doc = selected_doc_id
            st.rerun()

    with col3:
        if st.button("📊 Analyze", use_container_width=True):
            st.session_state.page = "analysis"
            st.session_state.selected_doc = selected_doc_id
            st.rerun()


def show_document_detail(doc, storage):
    """Show detailed document view"""
    doc_id = doc.get('doc_id')

    st.markdown("---")
    st.markdown(f"## 📄 {doc.get('filename', 'Unknown')}")

    # Metadata
    with st.expander("📋 Metadata", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Doc ID:** `{doc_id}`")
            st.write(f"**Status:** {doc.get('status', 'unknown')}")
            st.write(f"**Type:** {doc.get('doc_type', 'unknown')}")
            st.write(f"**Upload Date:** {doc.get('upload_date', 'N/A')}")

        with col2:
            if doc.get('page_count'):
                st.write(f"**Pages:** {doc['page_count']}")
            if doc.get('word_count'):
                st.write(f"**Words:** {doc['word_count']}")
            if doc.get('chunk_count'):
                st.write(f"**Chunks:** {doc['chunk_count']}")
            if doc.get('process_time'):
                st.write(f"**Process Time:** {doc['process_time']:.2f}s")

    # Text content
    try:
        text = storage.get_processed_text(doc_id)

        if text:
            with st.expander("📝 Extracted Text", expanded=False):
                st.text_area(
                    "Document Text",
                    text,
                    height=300,
                    disabled=True,
                    label_visibility="collapsed"
                )

                # Download button
                st.download_button(
                    "⬇️ Download Text",
                    text,
                    file_name=f"{doc.get('filename', 'document')}.txt",
                    mime="text/plain"
                )
    except:
        pass

    # Chunks
    try:
        chunks = storage.get_chunks(doc_id)

        if chunks:
            with st.expander(f"🧩 Chunks ({len(chunks)})", expanded=False):
                for i, chunk in enumerate(chunks[:10], 1):  # Show first 10
                    st.markdown(f"**Chunk {i}**")
                    st.code(chunk.get('text', '')[:200] + "...")
                    st.caption(f"Chunk ID: {chunk.get('chunk_id')}")
                    st.markdown("---")

                if len(chunks) > 10:
                    st.info(f"Showing first 10 of {len(chunks)} chunks")
    except:
        pass


if __name__ == "__main__":
    show_documents_page()
