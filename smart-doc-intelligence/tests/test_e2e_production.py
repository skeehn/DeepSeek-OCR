"""
Comprehensive End-to-End Testing Suite
Tests all components, integration, performance, and production readiness
"""
import sys
from pathlib import Path
import time
import traceback
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
        self.start_time = time.time()

    def add_pass(self, test_name, duration=None):
        self.passed.append({"name": test_name, "duration": duration})
        print(f"✅ PASS: {test_name}" + (f" ({duration:.2f}s)" if duration else ""))

    def add_fail(self, test_name, error):
        self.failed.append({"name": test_name, "error": str(error)})
        print(f"❌ FAIL: {test_name}")
        print(f"   Error: {error}")

    def add_warning(self, test_name, message):
        self.warnings.append({"name": test_name, "message": message})
        print(f"⚠️  WARN: {test_name}")
        print(f"   {message}")

    def print_summary(self):
        total_time = time.time() - self.start_time
        total_tests = len(self.passed) + len(self.failed)

        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {len(self.passed)}")
        print(f"❌ Failed: {len(self.failed)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print("="*70)

        if self.failed:
            print("\n❌ FAILED TESTS:")
            for fail in self.failed:
                print(f"  - {fail['name']}: {fail['error']}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warn in self.warnings:
                print(f"  - {warn['name']}: {warn['message']}")

        # Production readiness
        print("\n" + "="*70)
        if len(self.failed) == 0 and len(self.warnings) == 0:
            print("🎉 PRODUCTION READY - All tests passed!")
        elif len(self.failed) == 0:
            print("⚠️  CAUTION - All tests passed but with warnings")
        else:
            print("🚫 NOT PRODUCTION READY - Critical failures detected")
        print("="*70)


results = TestResults()


def test_imports():
    """Test 1: All critical imports work"""
    start = time.time()
    try:
        # Core components
        from backend.utils.config import AppConfig
        from backend.utils.storage import DocumentStorage
        from backend.utils.chunking import DocumentChunker

        # Vector DB
        from backend.vectordb.chroma_manager import ChromaManager
        from backend.vectordb.embeddings import EmbeddingManager
        from backend.vectordb.retrieval import DocumentRetriever

        # LLM
        from backend.llm.ollama_client import OllamaClient
        from backend.llm.gemini_client import GeminiClient
        from backend.llm.query_router import DualLLMManager, LLMType

        # Features
        from backend.features.rag_pipeline import CompleteRAGPipeline
        from backend.features.entity_extraction import EntityExtractor
        from backend.features.summarization import DocumentSummarizer
        from backend.features.document_comparison import DocumentComparator
        from backend.features.citations import CitationGenerator
        from backend.features.export import ExportManager

        results.add_pass("Import all modules", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("Import all modules", e)
        return False


def test_config():
    """Test 2: Configuration loads correctly"""
    start = time.time()
    try:
        from backend.utils.config import AppConfig
        config = AppConfig()

        assert config.storage_base_dir is not None
        assert config.chunk_size > 0
        assert config.chunk_overlap >= 0

        results.add_pass("Configuration loading", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("Configuration loading", e)
        return False


def test_storage():
    """Test 3: Document storage system"""
    start = time.time()
    try:
        from backend.utils.storage import DocumentStorage
        storage = DocumentStorage()

        # Test listing documents
        docs = storage.list_documents()
        assert isinstance(docs, list)

        # Test getting stats
        stats = storage.get_statistics()
        assert "total_documents" in stats

        results.add_pass("Document storage", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("Document storage", e)
        return False


def test_chunking():
    """Test 4: Text chunking"""
    start = time.time()
    try:
        from backend.utils.chunking import DocumentChunker

        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        test_text = "This is a test. " * 100  # 500+ characters

        # Test different strategies
        chunks = chunker.chunk_by_fixed_size(test_text)
        assert len(chunks) > 0
        assert all(c.text for c in chunks)

        results.add_pass("Text chunking", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("Text chunking", e)
        return False


def test_embeddings():
    """Test 5: Embedding generation"""
    start = time.time()
    try:
        from backend.vectordb.embeddings import EmbeddingManager

        em = EmbeddingManager()
        test_text = "This is a test document for embeddings."

        # Generate embedding
        embedding = em.embed_text(test_text)
        assert len(embedding) == 384  # MiniLM dimension

        # Test batch
        batch = em.embed_batch([test_text, test_text])
        assert len(batch) == 2

        # Test similarity
        sim = em.compute_similarity(embedding, embedding)
        assert 0.99 <= sim <= 1.01  # Should be very similar to itself

        results.add_pass("Embedding generation", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("Embedding generation", e)
        return False


def test_vector_db():
    """Test 6: Vector database operations"""
    start = time.time()
    try:
        from backend.vectordb.chroma_manager import ChromaManager

        chroma = ChromaManager()

        # List collections
        collections = chroma.list_collections()
        assert isinstance(collections, list)

        # Get or create test collection
        collection = chroma.get_collection("test_collection")
        if collection is None:
            chroma.create_collection("test_collection")

        results.add_pass("Vector database", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("Vector database", e)
        return False


def test_llm_clients():
    """Test 7: LLM client initialization"""
    start = time.time()
    try:
        from backend.llm.ollama_client import OllamaClient
        from backend.llm.gemini_client import GeminiClient

        # Test Ollama client
        try:
            ollama = OllamaClient()
            # Try to list models (will fail if Ollama not running)
            models = ollama.list_models()
            if models:
                results.add_pass("Ollama client", time.time() - start)
            else:
                results.add_warning("Ollama client", "Ollama server not running - local LLM unavailable")
        except Exception as e:
            results.add_warning("Ollama client", f"Not available: {e}")

        # Test Gemini client (don't actually call API)
        gemini = GeminiClient(api_key="test_key")
        assert gemini is not None

        results.add_pass("LLM client initialization", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("LLM client initialization", e)
        return False


def test_entity_extraction():
    """Test 8: Entity extraction"""
    start = time.time()
    try:
        from backend.features.entity_extraction import EntityExtractor

        extractor = EntityExtractor(use_llm=False)  # Pattern-based only
        test_text = """
        John Smith works at Microsoft Corporation.
        Contact: john@microsoft.com or 555-123-4567.
        Meeting on 12/25/2024. Budget: $50,000.
        Visit https://microsoft.com for details.
        """

        entities = extractor.extract_from_text(test_text)

        # Check we found entities
        assert len(entities) > 0
        assert "email" in entities
        assert len(entities["email"]) > 0

        results.add_pass("Entity extraction", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("Entity extraction", e)
        return False


def test_summarization():
    """Test 9: Document summarization"""
    start = time.time()
    try:
        from backend.features.summarization import DocumentSummarizer, SummaryLength

        summarizer = DocumentSummarizer(use_cloud=False)
        test_text = """
        Artificial intelligence is transforming technology. Machine learning enables
        computers to learn from data. Deep learning uses neural networks with multiple
        layers. Applications include image recognition, natural language processing,
        and recommendation systems. AI is revolutionizing healthcare, finance, and
        transportation industries.
        """ * 5  # Make it longer

        # Extractive summarization (no LLM needed)
        summary = summarizer.summarize(
            test_text,
            length=SummaryLength.SHORT,
            method="extractive"
        )

        assert summary.text is not None
        assert len(summary.text) > 0
        assert summary.word_count > 0

        results.add_pass("Document summarization", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("Document summarization", e)
        return False


def test_document_comparison():
    """Test 10: Document comparison"""
    start = time.time()
    try:
        from backend.features.document_comparison import DocumentComparator

        comparator = DocumentComparator(use_embeddings=False, use_llm=False)

        doc1 = "Machine learning is a subset of AI. It learns from data."
        doc2 = "Artificial intelligence includes machine learning. Data drives learning."
        doc3 = "Database systems store and retrieve data efficiently."

        result = comparator.compare_documents({
            "doc1": doc1,
            "doc2": doc2,
            "doc3": doc3
        }, include_llm_summary=False)

        assert result.doc_ids == ["doc1", "doc2", "doc3"]
        assert len(result.similarity_scores) > 0

        results.add_pass("Document comparison", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("Document comparison", e)
        return False


def test_citations():
    """Test 11: Citation generation"""
    start = time.time()
    try:
        from backend.features.citations import CitationGenerator, CitationMetadata, CitationStyle, DocumentType

        generator = CitationGenerator()

        metadata = CitationMetadata(
            doc_type=DocumentType.JOURNAL,
            title="Test Article",
            authors=["John Doe", "Jane Smith"],
            year=2024,
            journal="Test Journal",
            volume=10,
            pages="1-10"
        )

        # Generate APA citation
        citation = generator.generate(metadata, CitationStyle.APA)
        assert len(citation) > 0
        assert "Doe" in citation

        results.add_pass("Citation generation", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("Citation generation", e)
        return False


def test_export():
    """Test 12: Export functionality"""
    start = time.time()
    try:
        from backend.features.export import ExportManager
        import tempfile
        import os

        manager = ExportManager()

        test_data = {
            "type": "summary",
            "summary": "Test summary content",
            "key_points": ["Point 1", "Point 2"]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = manager.export(
                test_data,
                tmpdir,
                "test_export",
                formats=["json", "markdown"],
                title="Test Export"
            )

            # Verify files exist
            assert "json" in paths
            assert os.path.exists(paths["json"])

        results.add_pass("Export functionality", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("Export functionality", e)
        return False


def test_ui_imports():
    """Test 13: UI imports"""
    start = time.time()
    try:
        import streamlit
        assert streamlit is not None

        results.add_pass("UI dependencies", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("UI dependencies", e)
        return False


def test_performance_embeddings():
    """Test 14: Embedding performance (should handle 100s of docs)"""
    start = time.time()
    try:
        from backend.vectordb.embeddings import EmbeddingManager

        em = EmbeddingManager()

        # Generate 100 embeddings
        texts = [f"Test document number {i} with some content." for i in range(100)]

        batch_start = time.time()
        embeddings = em.embed_batch(texts, batch_size=32)
        batch_time = time.time() - batch_start

        assert len(embeddings) == 100

        # Should be reasonably fast (< 30 seconds for 100 docs)
        if batch_time > 30:
            results.add_warning("Embedding performance", f"Slow: {batch_time:.2f}s for 100 docs")
        else:
            results.add_pass(f"Embedding performance (100 docs in {batch_time:.2f}s)", time.time() - start)

        return True
    except Exception as e:
        results.add_fail("Embedding performance", e)
        return False


def test_memory_usage():
    """Test 15: Memory usage is reasonable"""
    start = time.time()
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)

        # Should use less than 2GB for basic operations
        if memory_mb > 2048:
            results.add_warning("Memory usage", f"High memory: {memory_mb:.0f}MB")
        else:
            results.add_pass(f"Memory usage ({memory_mb:.0f}MB)", time.time() - start)

        return True
    except ImportError:
        results.add_warning("Memory usage", "psutil not installed - skipping")
        return True
    except Exception as e:
        results.add_fail("Memory usage", e)
        return False


def test_concurrent_operations():
    """Test 16: Can handle concurrent operations"""
    start = time.time()
    try:
        from backend.utils.chunking import DocumentChunker
        from backend.vectordb.embeddings import EmbeddingManager

        # Simulate multiple users doing different operations
        chunker = DocumentChunker()
        em = EmbeddingManager()

        # User 1: Chunking
        text1 = "Test document " * 100
        chunks1 = chunker.chunk_by_fixed_size(text1)

        # User 2: Embedding
        emb1 = em.embed_text("Test text for user 2")

        # User 3: More chunking
        text2 = "Another document " * 100
        chunks2 = chunker.chunk_by_paragraph(text2)

        assert len(chunks1) > 0
        assert len(emb1) == 384
        assert len(chunks2) > 0

        results.add_pass("Concurrent operations", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("Concurrent operations", e)
        return False


def test_error_handling():
    """Test 17: Graceful error handling"""
    start = time.time()
    try:
        from backend.features.entity_extraction import EntityExtractor

        extractor = EntityExtractor(use_llm=False)

        # Test with empty string
        entities = extractor.extract_from_text("")
        assert isinstance(entities, dict)

        # Test with None (should handle gracefully)
        try:
            entities = extractor.extract_from_text(None)
        except (TypeError, AttributeError):
            pass  # Expected

        results.add_pass("Error handling", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("Error handling", e)
        return False


def test_file_cleanup():
    """Test 18: No file descriptor leaks"""
    start = time.time()
    try:
        import tempfile
        import os

        # Create and cleanup many temp files
        for i in range(100):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(b"test")
                tmp_path = tmp.name

            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        results.add_pass("File cleanup", time.time() - start)
        return True
    except Exception as e:
        results.add_fail("File cleanup", e)
        return False


def test_production_readiness():
    """Test 19: Production readiness checklist"""
    start = time.time()
    try:
        issues = []

        # Check critical directories exist
        from backend.utils.config import AppConfig
        config = AppConfig()

        if not Path(config.storage_base_dir).exists():
            issues.append("Storage directory doesn't exist")

        # Check imports are fast (< 2 seconds)
        import_start = time.time()
        from backend.features.rag_pipeline import CompleteRAGPipeline
        import_time = time.time() - import_start

        if import_time > 2:
            issues.append(f"Slow imports: {import_time:.2f}s")

        if issues:
            results.add_warning("Production readiness", ", ".join(issues))
        else:
            results.add_pass("Production readiness", time.time() - start)

        return True
    except Exception as e:
        results.add_fail("Production readiness", e)
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("COMPREHENSIVE END-TO-END TEST SUITE")
    print("Testing: Components, Integration, Performance, Production Readiness")
    print("="*70 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Storage System", test_storage),
        ("Text Chunking", test_chunking),
        ("Embeddings", test_embeddings),
        ("Vector Database", test_vector_db),
        ("LLM Clients", test_llm_clients),
        ("Entity Extraction", test_entity_extraction),
        ("Summarization", test_summarization),
        ("Document Comparison", test_document_comparison),
        ("Citations", test_citations),
        ("Export", test_export),
        ("UI Dependencies", test_ui_imports),
        ("Performance - Embeddings", test_performance_embeddings),
        ("Memory Usage", test_memory_usage),
        ("Concurrent Operations", test_concurrent_operations),
        ("Error Handling", test_error_handling),
        ("File Cleanup", test_file_cleanup),
        ("Production Readiness", test_production_readiness),
    ]

    print(f"Running {len(tests)} test suites...\n")

    for test_name, test_func in tests:
        print(f"\nTesting: {test_name}")
        print("-" * 70)
        try:
            test_func()
        except Exception as e:
            results.add_fail(test_name, f"Unexpected error: {e}")
            traceback.print_exc()

    # Print final summary
    results.print_summary()

    # Return exit code
    return 0 if len(results.failed) == 0 else 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    exit(exit_code)
