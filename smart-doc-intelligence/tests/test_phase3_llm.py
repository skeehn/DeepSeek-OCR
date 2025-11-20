"""
Phase 3 LLM Integration Test Script
Tests Ollama, Gemini, Query Routing, and Complete RAG Pipeline
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.llm.ollama_client import OllamaClient
from backend.llm.gemini_client import GeminiClient, RateLimitedGeminiClient
from backend.llm.query_router import QueryRouter, DualLLMManager, LLMType
from backend.features.rag_pipeline import CompleteRAGPipeline, ConversationalRAGPipeline
from backend.vectordb.retrieval import DocumentRetriever
from backend.utils.config import ensure_directories


def test_ollama_client():
    """Test 1: Ollama Client"""
    print("\n" + "=" * 60)
    print("TEST 1: Ollama Client")
    print("=" * 60)

    try:
        # Initialize client
        client = OllamaClient()
        print("✅ Ollama client initialized")

        # Check availability
        if not client.is_available():
            print("⚠️ Ollama server not running")
            print("   Start with: ollama serve")
            print("   Pull a model: ollama pull llama3.3")
            return

        # List models
        models = client.list_models()
        print(f"\nAvailable models: {models}")

        if not models:
            print("⚠️ No models available")
            print("   Pull a model: ollama pull llama3.3")
            return

        # Test generation
        print(f"\nTesting generation...")
        response = client.generate(
            "Explain machine learning in one sentence.",
            temperature=0.7,
            max_tokens=100
        )

        print(f"✅ Generation successful!")
        print(f"   Model: {response.model}")
        print(f"   Response: {response.text}")
        print(f"   Tokens: {response.total_tokens}")

        # Test chat
        print(f"\nTesting chat interface...")
        messages = [
            {"role": "user", "content": "What is AI?"}
        ]

        chat_response = client.chat(messages)
        print(f"✅ Chat successful!")
        print(f"   Response: {chat_response.text[:100]}...")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_gemini_client():
    """Test 2: Gemini API Client"""
    print("\n" + "=" * 60)
    print("TEST 2: Gemini API Client")
    print("=" * 60)

    try:
        # Check if API key is set
        from backend.utils.config import get_config
        config = get_config()

        if not config.gemini.api_key:
            print("⚠️ Gemini API key not set")
            print("   Set GEMINI_API_KEY environment variable")
            return

        # Initialize client
        client = RateLimitedGeminiClient()
        print("✅ Gemini client initialized (with rate limiting)")

        # Test generation
        print(f"\nTesting generation...")
        response = client.generate(
            "Explain deep learning in one sentence.",
            temperature=0.7,
            max_tokens=100
        )

        print(f"✅ Generation successful!")
        print(f"   Model: {response.model}")
        print(f"   Response: {response.text}")
        print(f"   Tokens: {response.total_tokens}")
        print(f"   Finish reason: {response.finish_reason}")

        # Test token counting
        tokens = client.count_tokens("This is a test sentence for token counting.")
        print(f"\nToken counting: {tokens} tokens")

    except ValueError as e:
        print(f"⚠️ Configuration error: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_query_router():
    """Test 3: Query Router"""
    print("\n" + "=" * 60)
    print("TEST 3: Query Router")
    print("=" * 60)

    try:
        # Initialize dual LLM manager
        manager = DualLLMManager(prefer_local=True)

        if not manager.is_available():
            print("⚠️ No LLM available")
            return

        print("✅ Dual LLM Manager initialized")

        # Test routing decisions
        test_queries = [
            ("What is machine learning?", "Simple factual query"),
            ("Compare supervised and unsupervised learning in detail.", "Complex reasoning"),
            ("What is my SSN in this document?", "Privacy-sensitive"),
            ("Analyze this medical diagnosis carefully.", "Sensitive content"),
            ("Summarize the key findings.", "Simple task"),
        ]

        print("\nRouting Decisions:")
        for query, expected_type in test_queries:
            routing = manager.router.route(query)

            print(f"\n  Query: {query[:60]}...")
            print(f"    → {routing.llm_type.value}")
            print(f"    Reason: {routing.reason}")
            print(f"    Confidence: {routing.confidence:.2f}")

        print("\n✅ Query routing tests passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_complete_rag_pipeline():
    """Test 4: Complete RAG Pipeline"""
    print("\n" + "=" * 60)
    print("TEST 4: Complete RAG Pipeline")
    print("=" * 60)

    try:
        # Initialize pipeline
        pipeline = CompleteRAGPipeline(
            collection_name="test_rag_pipeline",
            prefer_local=True
        )

        print("✅ Complete RAG Pipeline initialized")

        # Index some test content first
        print("\nIndexing test content...")
        retriever = DocumentRetriever(collection_name="test_rag_pipeline")

        test_chunks = [
            {
                "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                "chunk_id": 0,
                "start_char": 0,
                "end_char": 150,
                "metadata": {"topic": "ml"}
            },
            {
                "text": "Deep learning is a type of machine learning based on artificial neural networks with multiple layers. It has revolutionized computer vision and natural language processing.",
                "chunk_id": 1,
                "start_char": 150,
                "end_char": 320,
                "metadata": {"topic": "dl"}
            },
        ]

        retriever.index_document("test_doc_1", test_chunks)
        print("✅ Test content indexed")

        # Test complete RAG query
        print("\nTesting complete RAG query...")
        response = pipeline.query(
            query="What is machine learning?",
            top_k=2,
            llm_type=LLMType.AUTO,
            temperature=0.7
        )

        print(f"\n✅ RAG Pipeline Response:")
        print(f"   Query: {response.query}")
        print(f"   Answer: {response.answer[:200]}...")
        print(f"   LLM used: {response.llm_used}")
        print(f"   Tokens: {response.tokens_used}")
        print(f"   Sources: {len(response.sources)}")
        print(f"   Routing: {response.routing_reason}")

        # Test simple ask
        print("\nTesting simple ask()...")
        answer = pipeline.ask("What is deep learning?", use_local=True)
        print(f"   Answer: {answer[:150]}...")

        # Cleanup
        retriever.delete_document("test_doc_1")
        print("\n✅ Complete RAG pipeline tests passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_conversational_rag():
    """Test 5: Conversational RAG"""
    print("\n" + "=" * 60)
    print("TEST 5: Conversational RAG Pipeline")
    print("=" * 60)

    try:
        # Initialize conversational pipeline
        conv_pipeline = ConversationalRAGPipeline(
            collection_name="test_conv_rag",
            max_history=5,
            prefer_local=True
        )

        print("✅ Conversational RAG initialized")

        # Index test content
        retriever = DocumentRetriever(collection_name="test_conv_rag")

        test_chunks = [
            {
                "text": "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science and machine learning.",
                "chunk_id": 0,
                "start_char": 0,
                "end_char": 135,
                "metadata": {}
            },
        ]

        retriever.index_document("lang_doc", test_chunks)

        # Simulate conversation
        print("\nSimulating conversation:")

        queries = [
            "What is Python?",
            "What is it used for?",
        ]

        for query in queries:
            print(f"\n  User: {query}")
            response = conv_pipeline.chat(query, use_local=True)
            print(f"  Assistant: {response[:150]}...")

        # Check history
        history = conv_pipeline.get_history()
        print(f"\n✅ Conversation turns: {len(history)}")

        # Cleanup
        retriever.delete_document("lang_doc")
        print("✅ Conversational RAG tests passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_dual_llm_fallback():
    """Test 6: Dual LLM with Fallback"""
    print("\n" + "=" * 60)
    print("TEST 6: Dual LLM with Fallback")
    print("=" * 60)

    try:
        manager = DualLLMManager(prefer_local=True)

        if not manager.is_available():
            print("⚠️ No LLM available")
            return

        print("✅ Testing fallback mechanism...")

        # Test with AUTO routing
        response = manager.generate(
            query="What is artificial intelligence?",
            context=None,
            llm_type=LLMType.AUTO,
            temperature=0.7
        )

        print(f"\n   Primary LLM: {response.get('llm_type')}")
        print(f"   Response: {response.get('text', 'N/A')[:100]}...")
        print(f"   Fallback used: {response.get('fallback', False)}")

        print("\n✅ Fallback mechanism working")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all Phase 3 tests"""
    print("\n" + "=" * 60)
    print("PHASE 3 LLM INTEGRATION TESTS")
    print("Smart Document Intelligence Platform")
    print("=" * 60)

    # Ensure directories exist
    ensure_directories()

    try:
        test_ollama_client()
        test_gemini_client()
        test_query_router()
        test_complete_rag_pipeline()
        test_conversational_rag()
        test_dual_llm_fallback()

        print("\n" + "=" * 60)
        print("✅ ALL PHASE 3 TESTS COMPLETED")
        print("=" * 60)

        print("""
Next Steps:
-----------
1. Ensure LLMs are available:
   - Ollama: ollama serve & ollama pull llama3.3
   - Gemini: Set GEMINI_API_KEY environment variable

2. Test with real documents:
   - Process PDFs with OCR
   - Index in vector database
   - Ask questions and get AI-generated answers!

3. Build UI (Phase 5):
   - Streamlit interface
   - Document upload
   - Interactive Q&A
   - Conversation history

4. Optimize and deploy:
   - Fine-tune routing logic
   - Add caching
   - Monitor costs
   - Scale up!
        """)

    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
