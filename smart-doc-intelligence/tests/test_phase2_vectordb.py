"""
Phase 2 Vector Database & RAG Test Script
Tests the complete vector database and RAG functionality
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.vectordb.chroma_manager import ChromaManager
from backend.vectordb.embeddings import EmbeddingGenerator
from backend.vectordb.retrieval import DocumentRetriever
from backend.features.rag_engine import RAGEngine, ConversationalRAG
from backend.pipeline import DocumentPipeline
from backend.utils.config import ensure_directories


def test_chroma_manager():
    """Test 1: ChromaDB Manager"""
    print("\n" + "=" * 60)
    print("TEST 1: ChromaDB Manager")
    print("=" * 60)

    try:
        # Initialize manager
        manager = ChromaManager()
        print("✅ ChromaDB manager initialized")

        # Create test collection
        collection_name = "test_phase2"
        collection = manager.get_or_create_collection(collection_name)
        print(f"✅ Collection created: {collection_name}")

        # Add test documents
        docs = [
            "Machine learning is a field of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing enables computers to understand text.",
        ]

        metadatas = [
            {"topic": "ml", "doc_id": "test_1"},
            {"topic": "dl", "doc_id": "test_2"},
            {"topic": "nlp", "doc_id": "test_3"},
        ]

        manager.add_documents(collection_name, docs, metadatas)
        print(f"✅ Added {len(docs)} documents")

        # Search
        results = manager.search(
            collection_name,
            query="neural networks",
            n_results=2
        )

        print(f"\nSearch Results:")
        for i, doc in enumerate(results['documents'][0]):
            print(f"  {i+1}. {doc[:80]}...")

        # Collection stats
        stats = manager.get_collection_stats(collection_name)
        print(f"\nCollection Stats:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Unique docs: {stats['unique_documents']}")

        # Cleanup
        manager.delete_collection(collection_name)
        print(f"\n✅ Test collection cleaned up")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_embeddings():
    """Test 2: Embedding Generation"""
    print("\n" + "=" * 60)
    print("TEST 2: Embedding Generation")
    print("=" * 60)

    try:
        # Initialize embedding generator (use CPU for testing)
        generator = EmbeddingGenerator(device="cpu")
        print("✅ Embedding generator initialized")

        # Test single embedding
        text = "This is a test document about machine learning."
        embedding = generator.embed_text(text)

        print(f"\nSingle Embedding:")
        print(f"  Text: {text}")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding dimension: {generator.dimension}")

        # Test batch embedding
        texts = [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
            "NLP deals with human language.",
        ]

        embeddings = generator.embed_batch(texts, show_progress=False)

        print(f"\nBatch Embedding:")
        print(f"  Number of texts: {len(texts)}")
        print(f"  Embeddings shape: {embeddings.shape}")

        # Test similarity
        sim = generator.compute_similarity(embeddings[0], embeddings[1])
        print(f"\nSimilarity:")
        print(f"  Text 1 vs Text 2: {sim:.4f}")

        # Model info
        info = generator.get_model_info()
        print(f"\nModel Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        print("\n✅ Embedding tests passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_document_retrieval():
    """Test 3: Document Retrieval"""
    print("\n" + "=" * 60)
    print("TEST 3: Document Retrieval")
    print("=" * 60)

    try:
        # Initialize retriever
        retriever = DocumentRetriever(collection_name="test_retrieval")
        print("✅ Document retriever initialized")

        # Create test chunks
        test_chunks = [
            {
                "text": "Machine learning is a method of data analysis that automates analytical model building.",
                "chunk_id": 0,
                "start_char": 0,
                "end_char": 85,
                "metadata": {"topic": "ml", "importance": "high"}
            },
            {
                "text": "Deep learning is part of a broader family of machine learning methods based on neural networks.",
                "chunk_id": 1,
                "start_char": 86,
                "end_char": 180,
                "metadata": {"topic": "dl", "importance": "high"}
            },
            {
                "text": "Natural language processing is a subfield of linguistics and artificial intelligence.",
                "chunk_id": 2,
                "start_char": 181,
                "end_char": 267,
                "metadata": {"topic": "nlp", "importance": "medium"}
            },
        ]

        # Index document
        doc_id = "test_doc_123"
        retriever.index_document(doc_id, test_chunks)
        print(f"✅ Indexed document: {doc_id}")

        # Search
        results = retriever.search("neural networks", top_k=2)

        print(f"\nSearch Results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result.score:.4f}")
            print(f"     Doc: {result.doc_id}, Chunk: {result.chunk_id}")
            print(f"     Text: {result.text[:80]}...")

        # Get context
        context = retriever.get_context_for_query(
            "machine learning methods",
            top_k=2,
            max_context_length=500
        )

        print(f"\nContext for RAG:")
        print(f"  Length: {len(context)} characters")
        print(f"  Preview: {context[:150]}...")

        # Get stats
        stats = retriever.get_statistics()
        print(f"\nRetrieval Stats:")
        print(f"  Collection: {stats['collection_name']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Unique docs: {stats['unique_documents']}")

        # Cleanup
        retriever.delete_document(doc_id)
        print(f"\n✅ Cleanup complete")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_rag_engine():
    """Test 4: RAG Engine"""
    print("\n" + "=" * 60)
    print("TEST 4: RAG Engine")
    print("=" * 60)

    try:
        # Initialize RAG engine
        rag_engine = RAGEngine(collection_name="test_rag")
        print("✅ RAG engine initialized")

        # Index some sample content first
        retriever = DocumentRetriever(collection_name="test_rag")

        sample_chunks = [
            {
                "text": "Python is a high-level programming language known for its simplicity and readability.",
                "chunk_id": 0,
                "start_char": 0,
                "end_char": 85,
                "metadata": {"language": "python"}
            },
            {
                "text": "JavaScript is the programming language of the web, used for frontend and backend development.",
                "chunk_id": 1,
                "start_char": 0,
                "end_char": 94,
                "metadata": {"language": "javascript"}
            },
        ]

        retriever.index_document("lang_doc_1", sample_chunks)
        print("✅ Sample content indexed")

        # Execute RAG query
        query_result = rag_engine.query(
            query_text="What is Python?",
            top_k=2
        )

        print(f"\nRAG Query Result:")
        print(f"  Query: {query_result['query']}")
        print(f"  Chunks retrieved: {query_result['metadata']['num_chunks_retrieved']}")
        print(f"  Context length: {query_result['metadata']['context_length']}")

        # Build prompt
        prompt = rag_engine.build_prompt(
            query="What is Python?",
            context=query_result['context']
        )

        print(f"\nGenerated Prompt:")
        print(f"  Length: {len(prompt)} characters")
        print(f"  Preview: {prompt[:200]}...")

        # Test conversational RAG
        print(f"\n--- Conversational RAG ---")
        conv_rag = ConversationalRAG(collection_name="test_rag")

        # Simulate conversation (without LLM for now)
        response = conv_rag.chat("Tell me about Python", top_k=2)
        print(f"  Q: Tell me about Python")
        print(f"  A: {response.answer}")
        print(f"  Sources: {len(response.sources)}")

        # Cleanup
        retriever.delete_document("lang_doc_1")
        print(f"\n✅ RAG engine tests passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_integrated_pipeline():
    """Test 5: Integrated Pipeline with Vector DB"""
    print("\n" + "=" * 60)
    print("TEST 5: Integrated Pipeline with Vector DB")
    print("=" * 60)

    try:
        # Initialize pipeline with vector DB enabled
        pipeline = DocumentPipeline(
            load_ocr_model=False,  # Don't load OCR for this test
            enable_vectordb=True,
            collection_name="test_pipeline"
        )

        print("✅ Pipeline initialized with vector DB")

        # Simulate document processing (without actual OCR)
        # In real usage, you would process actual PDFs/images

        print("\nSimulating document indexing...")

        # Manually create chunks and index them
        from backend.utils.chunking import DocumentChunker

        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)

        sample_text = """
# Introduction to AI

Artificial Intelligence (AI) is the simulation of human intelligence by machines.
It encompasses various subfields including machine learning and deep learning.

## Machine Learning

Machine learning is a subset of AI that enables systems to learn from data.
It uses algorithms to identify patterns and make decisions with minimal human intervention.

## Deep Learning

Deep learning is a specialized form of machine learning that uses neural networks.
These networks can have many layers, hence the term "deep" learning.
        """

        chunks = chunker.chunk_by_markdown_section(sample_text)
        chunk_dicts = [chunk.to_dict() for chunk in chunks]

        # Index directly
        doc_id = "ai_intro_doc"
        if pipeline.retriever:
            pipeline.retriever.index_document(doc_id, chunk_dicts)
            print(f"✅ Document indexed: {doc_id}")

            # Test search
            results = pipeline.search_documents(
                query="What is machine learning?",
                top_k=2
            )

            print(f"\nSearch Results:")
            for i, result in enumerate(results):
                print(f"  {i+1}. Score: {result['score']:.4f}")
                print(f"     Text: {result['text'][:80]}...")

            # Test query
            query_result = pipeline.query_document(
                query="Explain deep learning",
                top_k=2
            )

            print(f"\nQuery Result:")
            print(f"  Chunks: {query_result['metadata']['num_chunks_retrieved']}")
            print(f"  Context preview: {query_result['context'][:150]}...")

            # Get statistics
            stats = pipeline.get_statistics()
            print(f"\nPipeline Stats:")
            print(f"  Storage docs: {stats.get('total_documents', 0)}")
            if 'vectordb' in stats:
                print(f"  Vector DB chunks: {stats['vectordb']['total_chunks']}")
                print(f"  Vector DB docs: {stats['vectordb']['unique_documents']}")

            # Cleanup
            pipeline.retriever.delete_document(doc_id)
            print(f"\n✅ Integrated pipeline tests passed")
        else:
            print("⚠️ Vector DB not available")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all Phase 2 tests"""
    print("\n" + "=" * 60)
    print("PHASE 2 VECTOR DATABASE & RAG TESTS")
    print("Smart Document Intelligence Platform")
    print("=" * 60)

    # Ensure directories exist
    ensure_directories()

    try:
        test_chroma_manager()
        test_embeddings()
        test_document_retrieval()
        test_rag_engine()
        test_integrated_pipeline()

        print("\n" + "=" * 60)
        print("✅ ALL PHASE 2 TESTS COMPLETED")
        print("=" * 60)

        print("""
Next Steps:
-----------
1. Install missing dependencies (if any):
   pip install chromadb sentence-transformers

2. Test with real documents:
   - Process PDFs/images with OCR
   - Index in vector database
   - Query and retrieve relevant content

3. Integrate LLMs (Phase 3):
   - Ollama for local processing
   - Gemini for complex reasoning
   - Complete RAG pipeline

4. Build UI (Phase 5):
   - Streamlit interface
   - Document upload
   - Query interface
        """)

    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
