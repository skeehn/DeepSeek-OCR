"""
Phase 4 Testing: Advanced Features
Tests entity extraction, document comparison, summarization, export, and citations
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.features.entity_extraction import (
    EntityExtractor,
    DocumentAnalyzer
)
from backend.features.document_comparison import (
    DocumentComparator,
    SimilarityAnalyzer
)
from backend.features.summarization import (
    DocumentSummarizer,
    SummaryStyle,
    SummaryLength
)
from backend.features.export import ExportManager
from backend.features.citations import (
    CitationGenerator,
    CitationMetadata,
    CitationStyle,
    DocumentType
)


def print_section(title: str):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_entity_extraction():
    """Test entity extraction"""
    print_section("TEST 1: Entity Extraction")

    test_text = """
    John Smith works at Microsoft Corporation in New York.
    He can be reached at john.smith@microsoft.com or 555-123-4567.
    The meeting is scheduled for 12/25/2024 at 3:00 PM.
    The project budget is $50,000 with a 15% contingency.
    Visit https://microsoft.com for more information.

    The team includes Jane Doe from Google and Bob Johnson from Apple.
    They will present at the AI Conference in San Francisco.
    """

    try:
        # Test entity extractor (without LLM for speed)
        print("1. Testing EntityExtractor (pattern-based)...")
        extractor = EntityExtractor(use_llm=False)
        entities = extractor.extract_from_text(test_text)

        print(f"\n✅ Extracted {len(entities)} entity types:")
        for entity_type, entity_list in entities.items():
            print(f"\n   {entity_type.upper()}:")
            for entity in entity_list[:5]:  # Show first 5
                print(f"     - {entity.text} (found {entity.occurrences} times)")

        # Test key terms
        print("\n2. Testing key terms extraction...")
        key_terms = extractor.extract_key_terms(test_text, top_k=10)
        print(f"\n✅ Top 10 key terms:")
        for term, freq in key_terms:
            print(f"     - {term}: {freq}")

        # Test document analyzer
        print("\n3. Testing DocumentAnalyzer...")
        analyzer = DocumentAnalyzer()
        analysis = analyzer.analyze_document(
            test_text,
            extract_entities=True,
            extract_key_terms=True
        )

        print(f"\n✅ Document analysis complete:")
        print(f"     Word count: {analysis['word_count']}")
        print(f"     Entity types: {len(analysis.get('entities', {}))}")
        print(f"     Key terms: {len(analysis.get('key_terms', []))}")

        print("\n✅ Entity extraction tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Entity extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_comparison():
    """Test document comparison"""
    print_section("TEST 2: Document Comparison")

    test_docs = {
        "doc1": """
        Machine learning is a subset of artificial intelligence that focuses on
        learning from data. It uses algorithms to identify patterns and make predictions.
        Common applications include image recognition, natural language processing, and
        recommendation systems. Deep learning uses neural networks with multiple layers.
        """,
        "doc2": """
        Artificial intelligence encompasses machine learning and deep learning.
        Machine learning algorithms learn from data to make predictions and decisions.
        Popular use cases include computer vision, speech recognition, and
        personalized recommendations. Neural networks are inspired by the human brain.
        """,
        "doc3": """
        Database systems store and manage structured data efficiently.
        They use SQL queries for data retrieval and manipulation.
        Common database types include relational, NoSQL, and graph databases.
        Data integrity and ACID properties are crucial for database design.
        """
    }

    try:
        # Test document comparator
        print("1. Testing DocumentComparator...")
        comparator = DocumentComparator(use_embeddings=False, use_llm=False)
        result = comparator.compare_documents(test_docs, include_llm_summary=False)

        print(f"\n✅ Comparison complete:")
        print(f"     Documents: {', '.join(result.doc_ids)}")
        print(f"     Average similarity: {result.metadata['avg_similarity']:.2%}")

        print(f"\n   Similarity scores:")
        for (doc1, doc2), score in result.similarity_scores.items():
            print(f"     {doc1} vs {doc2}: {score:.2%}")

        print(f"\n   Top 5 common terms:")
        for term, count in result.common_terms[:5]:
            print(f"     - {term}: {count}")

        # Test diff
        print("\n2. Testing document diff...")
        diff = comparator.diff_documents(
            "doc1", test_docs["doc1"],
            "doc2", test_docs["doc2"]
        )

        print(f"\n✅ Diff complete:")
        print(f"     Additions: {len(diff.additions)}")
        print(f"     Deletions: {len(diff.deletions)}")
        print(f"     Modifications: {len(diff.modifications)}")
        print(f"     Unchanged: {diff.unchanged_ratio:.1%}")

        print("\n✅ Document comparison tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Document comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_summarization():
    """Test summarization"""
    print_section("TEST 3: Summarization")

    test_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to
    the natural intelligence displayed by humans and animals. Leading AI textbooks define
    the field as the study of "intelligent agents": any device that perceives its environment
    and takes actions that maximize its chance of successfully achieving its goals.

    Machine learning is a subset of AI that focuses on the ability of machines to receive
    data and learn for themselves, changing algorithms as they learn more about the
    information they are processing. Deep learning is a subset of machine learning that
    uses neural networks with multiple layers.

    Applications of AI include advanced web search engines, recommendation systems,
    understanding human speech, self-driving cars, automated decision-making, and
    competing at the highest level in strategic game systems. As machines become
    increasingly capable, tasks considered to require "intelligence" are often removed
    from the definition of AI, a phenomenon known as the AI effect.

    Recent developments in AI have led to significant advances in natural language
    processing, computer vision, and robotics. Large language models can now generate
    human-like text, while computer vision systems can identify objects with high accuracy.
    However, challenges remain in areas such as common sense reasoning, general intelligence,
    and ethical AI development.
    """

    try:
        # Test extractive summarization
        print("1. Testing extractive summarization...")
        summarizer = DocumentSummarizer(use_cloud=False)

        summary = summarizer.summarize(
            test_text,
            style=SummaryStyle.PARAGRAPH,
            length=SummaryLength.SHORT,
            method="extractive"
        )

        print(f"\n✅ Extractive summary generated:")
        print(f"     Method: {summary.method}")
        print(f"     Word count: {summary.word_count}")
        print(f"     Compression: {summary.compression_ratio:.1%}")
        print(f"\n   Summary:")
        print(f"   {summary.text[:200]}...")

        # Test different lengths
        print("\n2. Testing different summary lengths...")
        for length in [SummaryLength.VERY_SHORT, SummaryLength.MEDIUM]:
            summary = summarizer.summarize(
                test_text,
                length=length,
                method="extractive"
            )
            print(f"     {length.value}: {summary.word_count} words")

        print("\n✅ Summarization tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Summarization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_export():
    """Test export functionality"""
    print_section("TEST 4: Export Functionality")

    test_data = {
        "type": "summary",
        "summary": "This is a test summary of a document about artificial intelligence and machine learning.",
        "key_points": [
            "AI is transforming various industries",
            "Machine learning enables computers to learn from data",
            "Deep learning uses neural networks"
        ],
        "metadata": {
            "word_count": 150,
            "compression_ratio": 0.25,
            "method": "extractive"
        }
    }

    try:
        # Create output directory
        output_dir = "./test_exports"
        os.makedirs(output_dir, exist_ok=True)

        # Test export manager
        print("1. Testing ExportManager...")
        manager = ExportManager()

        paths = manager.export(
            test_data,
            output_dir,
            "test_phase4_summary",
            formats=["json", "markdown", "html"],
            title="Test Summary Export"
        )

        print(f"\n✅ Export complete:")
        for format_type, path in paths.items():
            file_size = os.path.getsize(path)
            print(f"     {format_type}: {path} ({file_size} bytes)")

        # Test RAG response export
        print("\n2. Testing RAG response export...")
        rag_data = {
            "type": "rag_response",
            "query": "What is machine learning?",
            "answer": "Machine learning is a subset of AI that enables computers to learn from data.",
            "sources": [
                {
                    "doc_id": "doc1",
                    "text": "Machine learning algorithms learn from data...",
                    "score": 0.85
                }
            ],
            "metadata": {
                "llm_used": "ollama",
                "tokens": 150
            }
        }

        rag_paths = manager.export(
            rag_data,
            output_dir,
            "test_rag_response",
            formats=["json", "markdown"],
            title="RAG Query Response"
        )

        print(f"\n✅ RAG export complete:")
        for format_type, path in rag_paths.items():
            print(f"     {format_type}: {path}")

        print("\n✅ Export tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_citations():
    """Test citation generation"""
    print_section("TEST 5: Citation Generation")

    try:
        # Test citation generator
        print("1. Testing CitationGenerator...")
        generator = CitationGenerator()

        # Journal article
        journal_metadata = CitationMetadata(
            doc_type=DocumentType.JOURNAL,
            title="Deep Learning for Natural Language Processing",
            authors=["John Smith", "Jane Doe", "Bob Johnson"],
            year=2023,
            journal="Journal of Artificial Intelligence",
            volume=45,
            issue=3,
            pages="123-145",
            doi="10.1234/jai.2023.001"
        )

        print("\n   Journal Article Citations:\n")
        print("   APA:")
        print(f"   {generator.generate(journal_metadata, CitationStyle.APA)}")

        print("\n   MLA:")
        print(f"   {generator.generate(journal_metadata, CitationStyle.MLA)}")

        print("\n   Chicago:")
        print(f"   {generator.generate(journal_metadata, CitationStyle.CHICAGO)}")

        # Book
        print("\n2. Testing book citation...")
        book_metadata = CitationMetadata(
            doc_type=DocumentType.BOOK,
            title="Introduction to Machine Learning",
            authors=["Alice Brown"],
            year=2022,
            publisher="Tech Press",
            city="Cambridge",
            edition="3rd"
        )

        print(f"\n   Book Citation (APA):")
        print(f"   {generator.generate(book_metadata, CitationStyle.APA)}")

        # Website
        print("\n3. Testing website citation...")
        website_metadata = CitationMetadata(
            doc_type=DocumentType.WEBSITE,
            title="Understanding Neural Networks",
            authors=["Charlie Davis"],
            year=2024,
            publisher="AI Education",
            url="https://example.com/neural-networks",
            access_date="January 15, 2024"
        )

        print(f"\n   Website Citation (MLA):")
        print(f"   {generator.generate(website_metadata, CitationStyle.MLA)}")

        # Multiple styles
        print("\n4. Testing multiple style generation...")
        citations = generator.generate_multiple_styles(
            journal_metadata,
            styles=[CitationStyle.APA, CitationStyle.MLA]
        )

        print(f"\n✅ Generated {len(citations)} citation styles")

        print("\n✅ Citation tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Citation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration of multiple features"""
    print_section("TEST 6: Integration Test")

    test_text = """
    The research team at Stanford University, led by Dr. Sarah Johnson, published
    groundbreaking findings in Nature journal (volume 589, pages 234-240).
    Their work on neural networks has significant implications for AI development.
    Contact: research@stanford.edu for more information.
    The study was funded with a budget of $2.5 million.
    """

    try:
        print("1. Extract entities...")
        extractor = EntityExtractor(use_llm=False)
        entities = extractor.extract_from_text(test_text)
        print(f"     Entities extracted: {sum(len(v) for v in entities.values())}")

        print("\n2. Generate summary...")
        summarizer = DocumentSummarizer(use_cloud=False)
        summary = summarizer.summarize(
            test_text,
            length=SummaryLength.VERY_SHORT,
            method="extractive"
        )
        print(f"     Summary generated: {summary.word_count} words")

        print("\n3. Create citation...")
        citation_metadata = CitationMetadata(
            doc_type=DocumentType.JOURNAL,
            title="Neural Network Advances",
            authors=["Sarah Johnson"],
            year=2024,
            journal="Nature",
            volume=589,
            pages="234-240"
        )
        generator = CitationGenerator()
        citation = generator.generate(citation_metadata, CitationStyle.APA)
        print(f"     Citation generated: {len(citation)} characters")

        print("\n4. Export results...")
        export_data = {
            "type": "summary",
            "summary": summary.text,
            "entities": {
                k: [e.to_dict() for e in v]
                for k, v in entities.items()
            },
            "citation": citation
        }

        output_dir = "./test_exports"
        os.makedirs(output_dir, exist_ok=True)

        manager = ExportManager()
        paths = manager.export(
            export_data,
            output_dir,
            "integration_test",
            formats=["json"],
            title="Integration Test Results"
        )
        print(f"     Exported to: {paths.get('json', 'N/A')}")

        print("\n✅ Integration test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 4 tests"""
    print("\n" + "="*60)
    print("  PHASE 4: ADVANCED FEATURES - TEST SUITE")
    print("="*60)
    print("\nTesting: Entity Extraction, Comparison, Summarization,")
    print("         Export, and Citations\n")

    results = {
        "Entity Extraction": test_entity_extraction(),
        "Document Comparison": test_document_comparison(),
        "Summarization": test_summarization(),
        "Export": test_export(),
        "Citations": test_citations(),
        "Integration": test_integration(),
    }

    # Print summary
    print_section("TEST SUMMARY")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")

    print(f"\n   Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All Phase 4 tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
