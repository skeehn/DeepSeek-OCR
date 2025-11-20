"""
Example Usage of Smart Document Intelligence Platform
Demonstrates the Phase 1 OCR Pipeline
"""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.pipeline import DocumentPipeline
from backend.utils.config import ensure_directories


def example_1_simple_image():
    """Example 1: Process a single image"""
    print("\n" + "=" * 60)
    print("Example 1: Simple Image Processing")
    print("=" * 60)

    # Initialize pipeline with OCR model loaded
    pipeline = DocumentPipeline(load_ocr_model=True)

    # Process an image
    # Replace with your actual image path
    image_path = "sample_document.jpg"

    result = pipeline.process_image(
        image_path,
        prompt_type="document",  # Convert to markdown
        enhance=True,             # Enhance image quality
        save_to_storage=True      # Save to storage system
    )

    if result["success"]:
        print(f"\n✅ Success!")
        print(f"Document ID: {result['doc_id']}")
        print(f"Text extracted: {result['text_length']} characters")
        print(f"Chunks created: {result['chunk_count']}")
        print(f"\nFirst 200 characters:")
        print(result['extracted_text'][:200])
    else:
        print(f"❌ Failed: {result['error']}")


def example_2_pdf_processing():
    """Example 2: Process a PDF document"""
    print("\n" + "=" * 60)
    print("Example 2: PDF Processing")
    print("=" * 60)

    pipeline = DocumentPipeline(load_ocr_model=True)

    # Process a PDF
    pdf_path = "sample_document.pdf"

    result = pipeline.process_pdf(
        pdf_path,
        prompt_type="document",
        save_to_storage=True
    )

    if result["success"]:
        print(f"\n✅ PDF processed successfully!")
        print(f"Document ID: {result['doc_id']}")
        print(f"Pages: {result['page_count']}")
        print(f"Total text: {result['text_length']} characters")
        print(f"Chunks: {result['chunk_count']}")

        # Show per-page results
        print(f"\nPer-page results:")
        for page_result in result['page_results']:
            if page_result['success']:
                print(f"  Page {page_result['page']}: {page_result['text_length']} chars")
            else:
                print(f"  Page {page_result['page']}: Failed - {page_result.get('error')}")


def example_3_batch_processing():
    """Example 3: Batch process multiple files"""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)

    pipeline = DocumentPipeline(load_ocr_model=True)

    # List of files to process
    files = [
        "document1.pdf",
        "scan1.jpg",
        "invoice.png",
        "contract.pdf",
    ]

    results = pipeline.process_batch(files, prompt_type="document")

    print(f"\n📊 Batch Results:")
    for i, result in enumerate(results):
        if result["success"]:
            print(f"  {i+1}. ✅ {Path(files[i]).name} - {result['chunk_count']} chunks")
        else:
            print(f"  {i+1}. ❌ {Path(files[i]).name} - {result.get('error')}")


def example_4_retrieve_documents():
    """Example 4: Retrieve processed documents"""
    print("\n" + "=" * 60)
    print("Example 4: Retrieve Processed Documents")
    print("=" * 60)

    pipeline = DocumentPipeline()

    # List all documents
    docs = pipeline.list_documents()

    print(f"\n📚 Total documents: {len(docs)}")
    print("\nRecent documents:")

    for doc in docs[:5]:  # Show first 5
        print(f"\n  Document ID: {doc['doc_id']}")
        print(f"  Filename: {doc['original_filename']}")
        print(f"  Status: {doc['status']}")
        print(f"  Uploaded: {doc['upload_time']}")

        if doc['processed']:
            print(f"  Text length: {doc.get('text_length', 'N/A')}")
            print(f"  Chunks: {doc.get('chunk_count', 'N/A')}")


def example_5_document_details():
    """Example 5: Get detailed information about a document"""
    print("\n" + "=" * 60)
    print("Example 5: Document Details")
    print("=" * 60)

    pipeline = DocumentPipeline()

    # Get a document ID (replace with actual ID)
    docs = pipeline.list_documents()

    if docs:
        doc_id = docs[0]['doc_id']

        # Get detailed info
        info = pipeline.get_document_info(doc_id)

        print(f"\n📄 Document: {info['original_filename']}")
        print(f"   ID: {info['doc_id']}")
        print(f"   Type: {info['doc_type']}")
        print(f"   Size: {info['file_size']} bytes")
        print(f"   Processed: {info['processed']}")

        if info['has_text']:
            print(f"\n   Text preview:")
            print(f"   {info['text_preview']}...")

        if info['has_chunks']:
            print(f"\n   Has chunks: Yes")


def example_6_storage_statistics():
    """Example 6: View storage statistics"""
    print("\n" + "=" * 60)
    print("Example 6: Storage Statistics")
    print("=" * 60)

    pipeline = DocumentPipeline()

    stats = pipeline.get_statistics()

    print("\n📊 Storage Statistics:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Processed: {stats['processed_documents']}")
    print(f"   Pending: {stats['pending_documents']}")
    print(f"   Total size: {stats['total_size_mb']} MB")
    print(f"\n   Directories:")
    print(f"   - Uploads: {stats['uploads_dir']}")
    print(f"   - Processed: {stats['processed_dir']}")


def example_7_custom_chunking():
    """Example 7: Custom chunking strategies"""
    print("\n" + "=" * 60)
    print("Example 7: Custom Chunking Strategies")
    print("=" * 60)

    from backend.utils.chunking import DocumentChunker

    sample_text = """
# Introduction

This is a sample document with multiple sections.

## Section 1

Content for section 1 goes here. This section has important information.

## Section 2

More content in section 2. Different information here.

## Conclusion

Final thoughts and summary.
    """

    # Try different chunking strategies
    strategies = ["paragraph", "sentence", "markdown", "fixed"]

    for strategy in strategies:
        print(f"\n--- {strategy.upper()} Chunking ---")

        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(sample_text, strategy=strategy)

        print(f"Chunks created: {len(chunks)}")

        for i, chunk in enumerate(chunks[:2]):  # Show first 2
            print(f"\nChunk {i+1}:")
            print(f"  Length: {len(chunk.text)} chars")
            print(f"  Preview: {chunk.text[:80]}...")


def main():
    """Run examples"""
    print("\n" + "=" * 60)
    print("SMART DOCUMENT INTELLIGENCE PLATFORM")
    print("Example Usage Scripts")
    print("=" * 60)

    # Ensure directories exist
    ensure_directories()

    print("\nNOTE: These examples require:")
    print("  1. DeepSeek-OCR model downloaded")
    print("  2. GPU with sufficient VRAM")
    print("  3. Sample files (PDFs/images) to process")
    print("\nReplace sample file paths with your actual files.")

    # Uncomment the examples you want to run:

    # example_1_simple_image()
    # example_2_pdf_processing()
    # example_3_batch_processing()
    example_4_retrieve_documents()
    # example_5_document_details()
    example_6_storage_statistics()
    example_7_custom_chunking()


if __name__ == "__main__":
    main()
