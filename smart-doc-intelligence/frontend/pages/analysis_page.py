"""
Analysis Page
Document analysis tools: entities, comparison, summarization, export, citations
"""
import streamlit as st
from pathlib import Path
import sys
import json

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))


def show_analysis_page():
    """Display analysis page"""
    st.title("📊 Document Analysis")
    st.markdown("Advanced document intelligence features")

    # Analysis tabs
    tabs = st.tabs([
        "🔍 Entity Extraction",
        "📝 Summarization",
        "🔄 Document Comparison",
        "📚 Citations",
        "📤 Export"
    ])

    with tabs[0]:
        show_entity_extraction()

    with tabs[1]:
        show_summarization()

    with tabs[2]:
        show_comparison()

    with tabs[3]:
        show_citations()

    with tabs[4]:
        show_export()


def show_entity_extraction():
    """Entity extraction interface"""
    st.markdown("### 🔍 Entity Extraction")
    st.markdown("Extract named entities, key terms, and structured information from documents")

    # Document selection
    doc_id = select_document("Select document to analyze")

    if not doc_id:
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        # Entity types
        entity_types = st.multiselect(
            "Entity Types",
            ["email", "phone", "url", "date", "money", "percentage", "person", "organization", "location"],
            default=["email", "phone", "url", "date"]
        )

        extract_key_terms = st.checkbox("Extract Key Terms", value=True)

        use_llm = st.checkbox(
            "Use LLM for Entity Extraction",
            value=False,
            help="Use local LLM for advanced entity recognition (slower but more accurate)"
        )

    with col2:
        top_k_terms = st.slider("Number of Key Terms", 5, 50, 20)

    if st.button("🚀 Extract Entities", type="primary"):
        extract_entities(doc_id, entity_types, extract_key_terms, use_llm, top_k_terms)


def extract_entities(doc_id, entity_types, extract_key_terms, use_llm, top_k_terms):
    """Perform entity extraction"""
    from backend.utils.storage import DocumentStorage
    from backend.features.entity_extraction import EntityExtractor

    with st.spinner("Extracting entities..."):
        try:
            storage = DocumentStorage()
            text = storage.get_processed_text(doc_id)

            if not text:
                st.error("No text content found")
                return

            # Extract entities
            extractor = EntityExtractor(use_llm=use_llm)
            entities = extractor.extract_from_text(text, entity_types=entity_types or None)

            # Display results
            st.success("✅ Entity extraction complete!")

            # Summary
            total_entities = sum(len(v) for v in entities.values())
            st.metric("Total Entities Found", total_entities)

            # Show entities by type
            if entities:
                st.markdown("### Extracted Entities")

                for entity_type, entity_list in entities.items():
                    with st.expander(f"**{entity_type.upper()}** ({len(entity_list)} found)"):
                        for entity in entity_list[:20]:  # Show first 20
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"📌 **{entity.text}**")
                            with col2:
                                st.caption(f"{entity.occurrences} occurrences")

            # Key terms
            if extract_key_terms:
                st.markdown("### 🔑 Key Terms")
                key_terms = extractor.extract_key_terms(text, top_k=top_k_terms)

                if key_terms:
                    cols = st.columns(4)
                    for i, (term, freq) in enumerate(key_terms):
                        with cols[i % 4]:
                            st.metric(term, freq)

        except Exception as e:
            st.error(f"Error extracting entities: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


def show_summarization():
    """Summarization interface"""
    st.markdown("### 📝 Summarization")
    st.markdown("Generate summaries with different styles and lengths")

    doc_id = select_document("Select document to summarize")

    if not doc_id:
        return

    col1, col2 = st.columns(2)

    with col1:
        style = st.selectbox(
            "Summary Style",
            ["paragraph", "bullet_points", "executive", "technical", "simple"]
        )

        method = st.selectbox(
            "Method",
            ["auto", "extractive", "abstractive", "hierarchical"],
            help="auto: Automatically select best method\nextractive: Select key sentences\nabstractive: LLM-generated\nhierarchical: For long documents"
        )

    with col2:
        length = st.selectbox(
            "Summary Length",
            ["very_short", "short", "medium", "long", "detailed"]
        )

        use_cloud = st.checkbox(
            "Use Cloud LLM (Gemini)",
            value=False,
            help="Use Gemini for better quality (requires API key)"
        )

    if st.button("📝 Generate Summary", type="primary"):
        generate_summary(doc_id, style, length, method, use_cloud)


def generate_summary(doc_id, style, length, method, use_cloud):
    """Generate document summary"""
    from backend.utils.storage import DocumentStorage
    from backend.features.summarization import DocumentSummarizer, SummaryStyle, SummaryLength

    with st.spinner("Generating summary..."):
        try:
            storage = DocumentStorage()
            text = storage.get_processed_text(doc_id)

            if not text:
                st.error("No text content found")
                return

            # Map enum values
            style_enum = SummaryStyle(style)
            length_enum = SummaryLength(length)

            # Generate summary
            summarizer = DocumentSummarizer(use_cloud=use_cloud)
            summary = summarizer.summarize(
                text,
                style=style_enum,
                length=length_enum,
                method=None if method == "auto" else method
            )

            # Display results
            st.success("✅ Summary generated!")

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Method", summary.method)
            with col2:
                st.metric("Word Count", summary.word_count)
            with col3:
                st.metric("Compression", f"{summary.compression_ratio:.1%}")

            # Summary text
            st.markdown("### Summary")
            st.markdown(summary.text)

            # Key points
            if summary.key_points:
                with st.expander("🔑 Key Points"):
                    for point in summary.key_points:
                        st.write(f"- {point}")

            # Download
            st.download_button(
                "⬇️ Download Summary",
                summary.text,
                file_name=f"summary_{doc_id}.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Error generating summary: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


def show_comparison():
    """Document comparison interface"""
    st.markdown("### 🔄 Document Comparison")
    st.markdown("Compare multiple documents for similarity and differences")

    # Document selection
    docs = load_documents()

    if not docs:
        st.info("Upload at least 2 documents to compare")
        return

    selected_docs = st.multiselect(
        "Select 2-5 documents to compare",
        options=[d['doc_id'] for d in docs],
        format_func=lambda x: next((d['filename'] for d in docs if d['doc_id'] == x), x),
        max_selections=5
    )

    if len(selected_docs) < 2:
        st.warning("Please select at least 2 documents")
        return

    col1, col2 = st.columns(2)

    with col1:
        use_embeddings = st.checkbox("Use Semantic Similarity", value=True)

    with col2:
        include_llm_summary = st.checkbox("Generate LLM Summary", value=False)

    if st.button("🔄 Compare Documents", type="primary"):
        compare_documents(selected_docs, use_embeddings, include_llm_summary)


def compare_documents(doc_ids, use_embeddings, include_llm_summary):
    """Compare selected documents"""
    from backend.utils.storage import DocumentStorage
    from backend.features.document_comparison import DocumentComparator

    with st.spinner("Comparing documents..."):
        try:
            storage = DocumentStorage()

            # Load document texts
            documents = {}
            for doc_id in doc_ids:
                text = storage.get_processed_text(doc_id)
                if text:
                    documents[doc_id] = text

            if len(documents) < 2:
                st.error("Could not load enough documents")
                return

            # Compare
            comparator = DocumentComparator(use_embeddings=use_embeddings, use_llm=include_llm_summary)
            result = comparator.compare_documents(documents, include_llm_summary=include_llm_summary)

            # Display results
            st.success("✅ Comparison complete!")

            # Average similarity
            st.metric("Average Similarity", f"{result.metadata['avg_similarity']:.1%}")

            # Pairwise similarities
            st.markdown("### 📊 Pairwise Similarities")

            for (doc1, doc2), score in result.similarity_scores.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{doc1}** vs **{doc2}**")
                with col2:
                    st.progress(score, text=f"{score:.1%}")

            # Common terms
            if result.common_terms:
                with st.expander("🔤 Common Terms (Top 15)"):
                    cols = st.columns(3)
                    for i, (term, count) in enumerate(result.common_terms[:15]):
                        with cols[i % 3]:
                            st.metric(term, count)

            # LLM summary
            if result.summary:
                st.markdown("### 📝 Analysis Summary")
                st.markdown(result.summary)

        except Exception as e:
            st.error(f"Error comparing documents: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


def show_citations():
    """Citation generation interface"""
    st.markdown("### 📚 Citation Generator")
    st.markdown("Generate academic citations in APA, MLA, or Chicago style")

    # Citation style
    style = st.selectbox(
        "Citation Style",
        ["APA (7th Edition)", "MLA (9th Edition)", "Chicago (17th Edition)"]
    )

    # Document type
    doc_type = st.selectbox(
        "Document Type",
        ["journal", "book", "article", "website", "conference", "thesis", "report"]
    )

    # Citation form
    with st.form("citation_form"):
        st.markdown("#### Enter Citation Information")

        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input("Title *", help="Required")
            authors_str = st.text_input("Authors (comma-separated) *", help="e.g., John Smith, Jane Doe")
            year = st.number_input("Year", min_value=1900, max_value=2100, value=2024)

        with col2:
            if doc_type == "journal":
                journal = st.text_input("Journal Name")
                volume = st.number_input("Volume", min_value=0, value=0)
                issue = st.number_input("Issue", min_value=0, value=0)
                pages = st.text_input("Pages", help="e.g., 123-145")
                doi = st.text_input("DOI", help="e.g., 10.1234/journal.2024.001")

            elif doc_type == "book":
                publisher = st.text_input("Publisher")
                city = st.text_input("City")
                edition = st.text_input("Edition", help="e.g., 3rd")

            elif doc_type == "website":
                publisher = st.text_input("Website Name")
                url = st.text_input("URL")
                access_date = st.text_input("Access Date", help="e.g., January 15, 2024")

        submitted = st.form_submit_button("📚 Generate Citation", type="primary")

        if submitted:
            generate_citation(
                style, doc_type, title, authors_str, year,
                locals()
            )


def generate_citation(style, doc_type, title, authors_str, year, metadata):
    """Generate citation"""
    from backend.features.citations import CitationGenerator, CitationMetadata, CitationStyle, DocumentType

    try:
        if not title or not authors_str:
            st.error("Please provide title and authors")
            return

        # Parse authors
        authors = [a.strip() for a in authors_str.split(',')]

        # Map style
        style_map = {
            "APA (7th Edition)": CitationStyle.APA,
            "MLA (9th Edition)": CitationStyle.MLA,
            "Chicago (17th Edition)": CitationStyle.CHICAGO
        }

        # Create metadata
        citation_metadata = CitationMetadata(
            doc_type=DocumentType(doc_type),
            title=title,
            authors=authors,
            year=year,
            publisher=metadata.get('publisher'),
            journal=metadata.get('journal'),
            volume=metadata.get('volume') if metadata.get('volume', 0) > 0 else None,
            issue=metadata.get('issue') if metadata.get('issue', 0) > 0 else None,
            pages=metadata.get('pages'),
            doi=metadata.get('doi'),
            url=metadata.get('url'),
            access_date=metadata.get('access_date'),
            edition=metadata.get('edition'),
            city=metadata.get('city')
        )

        # Generate citation
        generator = CitationGenerator()
        citation = generator.generate(citation_metadata, style_map[style])

        # Display
        st.success("✅ Citation generated!")
        st.markdown("### Citation")
        st.code(citation, language=None)

        # Copy button
        st.download_button(
            "📋 Download Citation",
            citation,
            file_name="citation.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error generating citation: {e}")


def show_export():
    """Export interface"""
    st.markdown("### 📤 Export Results")
    st.markdown("Export analysis results to JSON, Markdown, or HTML")

    doc_id = select_document("Select document to export")

    if not doc_id:
        return

    # Export options
    col1, col2 = st.columns(2)

    with col1:
        export_formats = st.multiselect(
            "Export Formats",
            ["JSON", "Markdown", "HTML"],
            default=["JSON", "Markdown"]
        )

        include_entities = st.checkbox("Include Entity Extraction", value=True)
        include_summary = st.checkbox("Include Summary", value=True)

    with col2:
        title = st.text_input("Export Title", value="Document Analysis Results")

    if st.button("📤 Export", type="primary"):
        export_document(doc_id, export_formats, include_entities, include_summary, title)


def export_document(doc_id, formats, include_entities, include_summary, title):
    """Export document analysis"""
    from backend.utils.storage import DocumentStorage
    from backend.features.export import ExportManager
    from backend.features.entity_extraction import EntityExtractor
    from backend.features.summarization import DocumentSummarizer, SummaryLength

    with st.spinner("Exporting..."):
        try:
            storage = DocumentStorage()
            text = storage.get_processed_text(doc_id)

            if not text:
                st.error("No text content found")
                return

            # Prepare export data
            export_data = {
                "type": "analysis",
                "doc_id": doc_id,
                "title": title
            }

            # Add entities
            if include_entities:
                extractor = EntityExtractor(use_llm=False)
                entities = extractor.extract_from_text(text)
                export_data["entities"] = {
                    k: [e.to_dict() for e in v]
                    for k, v in entities.items()
                }

            # Add summary
            if include_summary:
                summarizer = DocumentSummarizer(use_cloud=False)
                summary = summarizer.summarize(text, length=SummaryLength.MEDIUM, method="extractive")
                export_data["summary"] = summary.to_dict()

            # Export
            manager = ExportManager()
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = manager.export(
                    export_data,
                    tmpdir,
                    f"analysis_{doc_id}",
                    formats=[f.lower() for f in formats],
                    title=title
                )

                st.success("✅ Export complete!")

                # Provide downloads
                for format_type, path in paths.items():
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    st.download_button(
                        f"⬇️ Download {format_type.upper()}",
                        content,
                        file_name=f"analysis_{doc_id}.{format_type}",
                        mime=get_mime_type(format_type)
                    )

        except Exception as e:
            st.error(f"Error exporting: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


# Helper functions

def select_document(label):
    """Document selector widget"""
    docs = load_documents()

    if not docs:
        st.info("No documents available. Upload documents first.")
        return None

    doc_id = st.selectbox(
        label,
        options=[d['doc_id'] for d in docs],
        format_func=lambda x: next((d['filename'] for d in docs if d['doc_id'] == x), x)
    )

    return doc_id


def load_documents():
    """Load all documents"""
    try:
        from backend.utils.storage import DocumentStorage
        storage = DocumentStorage()
        docs = storage.list_documents()
        return [d for d in docs if d.get('status') == 'processed']
    except:
        return []


def get_mime_type(format_type):
    """Get MIME type for format"""
    mime_types = {
        "json": "application/json",
        "markdown": "text/markdown",
        "html": "text/html"
    }
    return mime_types.get(format_type, "text/plain")


if __name__ == "__main__":
    show_analysis_page()
