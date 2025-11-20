"""
Phase 2 Example Usage
Demonstrates Vector Database and RAG features
"""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.pipeline import DocumentPipeline
from backend.vectordb.retrieval import DocumentRetriever
from backend.features.rag_engine import RAGEngine, ConversationalRAG
from backend.utils.config import ensure_directories


def example_1_vector_db_indexing():
    """Example 1: Document Processing with Vector DB Indexing"""
    print("\n" + "=" * 60)
    print("Example 1: Document Processing with Vector DB Indexing")
    print("=" * 60)

    # Initialize pipeline WITH vector database enabled
    pipeline = DocumentPipeline(
        load_ocr_model=True,
        enable_vectordb=True,  # Enable Phase 2 feature!
        collection_name="my_documents"
    )

    # Process a document (PDF or image)
    # The document will be:
    # 1. OCR'd to extract text
    # 2. Chunked into semantic pieces
    # 3. Indexed in ChromaDB for semantic search

    result = pipeline.process_pdf(
        "your_document.pdf",
        prompt_type="document",
        save_to_storage=True
    )

    if result["success"]:
        print(f"\n✅ Document processed and indexed!")
        print(f"   Doc ID: {result['doc_id']}")
        print(f"   Chunks: {result['chunk_count']}")
        print(f"   Indexed in vector database: ✅")


def example_2_semantic_search():
    """Example 2: Semantic Search Across Documents"""
    print("\n" + "=" * 60)
    print("Example 2: Semantic Search Across Documents")
    print("=" * 60)

    # Initialize pipeline with vector DB
    pipeline = DocumentPipeline(
        enable_vectordb=True,
        collection_name="my_documents"
    )

    # Search across all indexed documents
    results = pipeline.search_documents(
        query="What are the main findings about machine learning?",
        top_k=5  # Get top 5 most relevant chunks
    )

    print(f"\n🔍 Search Results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Relevance Score: {result['score']:.4f}")
        print(f"   From: {result['doc_id']}, Chunk {result['chunk_id']}")
        print(f"   Text: {result['text'][:150]}...")


def example_3_search_within_document():
    """Example 3: Search Within a Specific Document"""
    print("\n" + "=" * 60)
    print("Example 3: Search Within Specific Document")
    print("=" * 60)

    pipeline = DocumentPipeline(
        enable_vectordb=True,
        collection_name="my_documents"
    )

    # Get list of documents
    docs = pipeline.list_documents()
    if docs:
        doc_id = docs[0]['doc_id']  # Use first document

        # Search only within this document
        results = pipeline.search_documents(
            query="summary of key points",
            top_k=3,
            filter_doc_id=doc_id  # Filter by specific document
        )

        print(f"\n📄 Results from document {doc_id}:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['text'][:100]}...")


def example_4_rag_query():
    """Example 4: RAG Query - Get Context for LLM"""
    print("\n" + "=" * 60)
    print("Example 4: RAG Query (Retrieve Context for LLM)")
    print("=" * 60)

    pipeline = DocumentPipeline(
        enable_vectordb=True,
        collection_name="my_documents"
    )

    # Query documents and get relevant context
    # This prepares the context for feeding to an LLM
    query_result = pipeline.query_document(
        query="What are the benefits of deep learning?",
        top_k=5
    )

    print(f"\n📝 RAG Query Result:")
    print(f"   Query: {query_result['query']}")
    print(f"   Chunks retrieved: {query_result['metadata']['num_chunks_retrieved']}")
    print(f"   Context length: {query_result['metadata']['context_length']} chars")

    print(f"\n   Context preview:")
    print(f"   {query_result['context'][:300]}...")

    # This context can now be sent to an LLM (Ollama or Gemini)
    # along with the query to generate an answer


def example_5_direct_rag_engine():
    """Example 5: Using RAG Engine Directly"""
    print("\n" + "=" * 60)
    print("Example 5: Using RAG Engine Directly")
    print("=" * 60)

    # Create RAG engine
    rag_engine = RAGEngine(collection_name="my_documents")

    # Query
    query = "Explain the differences between supervised and unsupervised learning"

    result = rag_engine.query(
        query_text=query,
        top_k=5,
        max_context_length=2000
    )

    print(f"\n💡 RAG Query:")
    print(f"   Query: {query}")
    print(f"   Sources found: {len(result['sources'])}")

    # Show sources
    print(f"\n   Sources:")
    for i, source in enumerate(result['sources'][:3]):
        print(f"   {i+1}. Doc: {source.doc_id}, Chunk: {source.chunk_id}")
        print(f"      Score: {source.score:.4f}")
        print(f"      Text: {source.text[:100]}...")

    # Get the formatted context
    print(f"\n   Context ready for LLM: {len(result['context'])} characters")


def example_6_conversational_rag():
    """Example 6: Conversational RAG (Multi-turn)"""
    print("\n" + "=" * 60)
    print("Example 6: Conversational RAG")
    print("=" * 60)

    # Create conversational RAG
    conv_rag = ConversationalRAG(
        collection_name="my_documents",
        max_history=5  # Keep last 5 exchanges
    )

    # Simulate a conversation
    queries = [
        "What is machine learning?",
        "How is it different from traditional programming?",
        "What are some real-world applications?",
    ]

    print(f"\n💬 Conversation:")
    for query in queries:
        # Get response (without LLM for now, just retrieval)
        response = conv_rag.chat(query, top_k=3)

        print(f"\n   User: {query}")
        print(f"   System: Found {len(response.sources)} relevant passages")
        # In Phase 3, an LLM would generate the actual answer
        print(f"   Answer: {response.answer}")

    # View conversation history
    history = conv_rag.get_history()
    print(f"\n   Conversation turns: {len(history)}")


def example_7_document_comparison():
    """Example 7: Compare Multiple Documents"""
    print("\n" + "=" * 60)
    print("Example 7: Compare Multiple Documents")
    print("=" * 60)

    rag_engine = RAGEngine(collection_name="my_documents")

    # Get list of documents
    retriever = DocumentRetriever(collection_name="my_documents")

    # Assume we have multiple documents indexed
    doc_ids = ["doc_1", "doc_2", "doc_3"]  # Replace with actual doc IDs

    # Compare how each document addresses a topic
    comparison = rag_engine.compare_documents(
        query="machine learning applications",
        doc_ids=doc_ids,
        top_k_per_doc=2
    )

    print(f"\n📊 Document Comparison:")
    print(f"   Query: {comparison['query']}")

    for doc_id, doc_results in comparison['documents'].items():
        print(f"\n   Document: {doc_id}")
        print(f"     Best score: {doc_results['best_score']:.4f}")
        print(f"     Results: {doc_results['num_results']}")


def example_8_statistics_and_monitoring():
    """Example 8: Monitor Vector DB Statistics"""
    print("\n" + "=" * 60)
    print("Example 8: Statistics and Monitoring")
    print("=" * 60)

    pipeline = DocumentPipeline(
        enable_vectordb=True,
        collection_name="my_documents"
    )

    # Get comprehensive statistics
    stats = pipeline.get_statistics()

    print(f"\n📈 System Statistics:")

    # Storage stats
    print(f"\n   File Storage:")
    print(f"     Total documents: {stats.get('total_documents', 0)}")
    print(f"     Processed: {stats.get('processed_documents', 0)}")
    print(f"     Storage size: {stats.get('total_size_mb', 0)} MB")

    # Vector DB stats
    if 'vectordb' in stats:
        print(f"\n   Vector Database:")
        print(f"     Collection: {stats['vectordb']['collection_name']}")
        print(f"     Total chunks: {stats['vectordb']['total_chunks']}")
        print(f"     Unique documents: {stats['vectordb']['unique_documents']}")
        print(f"     Embedding model: {stats['vectordb']['embedding_model']}")


def example_9_advanced_search():
    """Example 9: Advanced Search with Filters"""
    print("\n" + "=" * 60)
    print("Example 9: Advanced Search with Metadata Filters")
    print("=" * 60)

    retriever = DocumentRetriever(collection_name="my_documents")

    # Search with metadata filter
    # (Assumes documents were indexed with custom metadata)

    # Example: Search only in documents tagged as "research papers"
    # This requires ChromaDB's where clause (coming in Phase 3)

    # For now, basic search
    results = retriever.search(
        query="neural network architectures",
        top_k=5,
        score_threshold=0.7  # Only results with score > 0.7
    )

    print(f"\n🎯 Advanced Search Results:")
    print(f"   Found: {len(results)} high-relevance results")

    for i, result in enumerate(results):
        print(f"\n   {i+1}. Score: {result.score:.4f}")
        print(f"      Doc: {result.doc_id}")
        print(f"      Text: {result.text[:120]}...")


def main():
    """Run examples"""
    print("\n" + "=" * 60)
    print("PHASE 2: VECTOR DATABASE & RAG EXAMPLES")
    print("Smart Document Intelligence Platform")
    print("=" * 60)

    # Ensure directories exist
    ensure_directories()

    print("\nNOTE: These examples require:")
    print("  1. ChromaDB and sentence-transformers installed")
    print("     pip install chromadb sentence-transformers")
    print("  2. Documents processed and indexed")
    print("  3. Sample files to process (for some examples)")
    print("\nUncomment the examples you want to run:")

    # Uncomment to run specific examples:

    # Process and index documents
    # example_1_vector_db_indexing()

    # Search examples
    # example_2_semantic_search()
    # example_3_search_within_document()

    # RAG examples
    # example_4_rag_query()
    # example_5_direct_rag_engine()
    example_6_conversational_rag()

    # Advanced features
    # example_7_document_comparison()
    example_8_statistics_and_monitoring()
    # example_9_advanced_search()


if __name__ == "__main__":
    main()
