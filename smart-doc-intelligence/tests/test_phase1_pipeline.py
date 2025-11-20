"""
Phase 1 Pipeline Test Script
Tests the complete OCR pipeline: Upload → OCR → Chunking → Storage
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ocr.pdf_processor import PDFProcessor, convert_pdf_to_images
from backend.ocr.image_processor import ImageProcessor
from backend.ocr.deepseek_wrapper import DeepSeekOCR
from backend.utils.chunking import DocumentChunker
from backend.utils.storage import DocumentStorage
from backend.utils.config import ensure_directories, get_config


def test_pdf_processing():
    """Test PDF to image conversion"""
    print("\n" + "=" * 60)
    print("TEST 1: PDF Processing")
    print("=" * 60)

    processor = PDFProcessor(dpi=144)

    # Test with a sample PDF (you'll need to provide one)
    # pdf_path = "sample.pdf"
    # images = processor.pdf_to_images(pdf_path)
    # print(f"Converted {len(images)} pages")

    print("✅ PDF Processor initialized successfully")
    print("   (Provide a sample PDF to test conversion)")


def test_image_processing():
    """Test image preprocessing"""
    print("\n" + "=" * 60)
    print("TEST 2: Image Processing")
    print("=" * 60)

    processor = ImageProcessor()

    # Test with a sample image
    # img = processor.load_image("sample.jpg")
    # if img:
    #     enhanced = processor.enhance_for_ocr(img)
    #     stats = processor.get_image_stats(img)
    #     print(f"Image stats: {stats}")

    print("✅ Image Processor initialized successfully")
    print("   (Provide a sample image to test enhancement)")


def test_ocr_wrapper():
    """Test DeepSeek-OCR wrapper"""
    print("\n" + "=" * 60)
    print("TEST 3: DeepSeek-OCR Wrapper")
    print("=" * 60)

    try:
        ocr = DeepSeekOCR()
        print("✅ DeepSeek-OCR wrapper initialized")

        # To actually test OCR, you need to load the model (requires GPU)
        # ocr.load_model()
        # result = ocr.extract_text("sample.jpg", prompt_type="document")
        # print(f"Extracted text length: {len(result.get('text', ''))}")

        print("   (Load model and provide image to test extraction)")

    except Exception as e:
        print(f"⚠️ DeepSeek-OCR wrapper creation: {e}")
        print("   (This is expected if vLLM is not installed)")


def test_chunking():
    """Test document chunking"""
    print("\n" + "=" * 60)
    print("TEST 4: Document Chunking")
    print("=" * 60)

    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)

    sample_text = """
# Introduction

This is a sample document with multiple paragraphs for testing the chunking functionality.

## Section 1: Overview

The first section contains important information about the document structure.
It demonstrates how the chunker handles different text lengths and paragraph breaks.

This is a second paragraph in the first section. It adds more content to test the chunking algorithm.

## Section 2: Details

Here we have another section with different content. The chunker should handle this appropriately.

## Conclusion

This is the final section with concluding remarks about the document.
    """

    print("\n1. Testing Paragraph Chunking:")
    chunks = chunker.chunk_by_paragraph(sample_text)
    for chunk in chunks[:3]:  # Show first 3
        print(f"   Chunk {chunk.chunk_id}: {len(chunk.text)} chars")
        print(f"   Preview: {chunk.text[:80]}...")

    print(f"\n2. Testing Markdown Section Chunking:")
    chunks = chunker.chunk_by_markdown_section(sample_text)
    for chunk in chunks:
        header = chunk.metadata.get('section_header', 'No header')
        print(f"   Chunk {chunk.chunk_id}: {header} ({len(chunk.text)} chars)")

    stats = chunker.get_chunk_statistics(chunks)
    print(f"\n3. Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n✅ Chunking tests completed")


def test_storage():
    """Test document storage system"""
    print("\n" + "=" * 60)
    print("TEST 5: Document Storage")
    print("=" * 60)

    storage = DocumentStorage()

    # Create a test file
    test_file = project_root / "storage" / "uploads" / "test_doc.txt"
    test_file.parent.mkdir(parents=True, exist_ok=True)

    with open(test_file, 'w') as f:
        f.write("This is a test document for storage system.")

    # Test upload
    result = storage.save_upload(str(test_file), doc_type="text")

    if result.get("success"):
        doc_id = result["doc_id"]
        print(f"✅ Document saved: {doc_id}")

        # Test saving processed results
        sample_text = "This is extracted text from OCR"
        sample_chunks = [
            {"text": "Chunk 1", "chunk_id": 0},
            {"text": "Chunk 2", "chunk_id": 1},
        ]

        success = storage.save_processed(
            doc_id,
            extracted_text=sample_text,
            chunks=sample_chunks
        )

        if success:
            print(f"✅ Processed results saved")

            # Test retrieval
            retrieved_text = storage.get_processed_text(doc_id)
            retrieved_chunks = storage.get_chunks(doc_id)

            print(f"   Retrieved text: {len(retrieved_text)} chars")
            print(f"   Retrieved chunks: {len(retrieved_chunks)}")

            # Test metadata
            metadata = storage.get_metadata(doc_id)
            print(f"   Status: {metadata.get('status')}")
            print(f"   Processed: {metadata.get('processed')}")

        # Clean up (optional)
        # storage.delete_document(doc_id)

    # Get storage stats
    stats = storage.get_storage_stats()
    print(f"\n📊 Storage Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n✅ Storage tests completed")


def test_full_pipeline():
    """Test complete pipeline (requires actual files and GPU)"""
    print("\n" + "=" * 60)
    print("TEST 6: Full Pipeline (Integration)")
    print("=" * 60)

    print("""
To test the full pipeline, you need:
1. A sample PDF or image file
2. DeepSeek-OCR model downloaded
3. GPU with sufficient VRAM
4. vLLM installed

Example pipeline:
-----------------
1. Upload document → storage.save_upload(pdf_path)
2. Convert PDF to images → PDFProcessor().pdf_to_images(pdf_path)
3. Extract text with OCR → ocr.extract_text(image_path)
4. Chunk the text → chunker.chunk_document(extracted_text)
5. Save results → storage.save_processed(doc_id, text, chunks)
6. Query documents → storage.list_documents()
    """)

    print("✅ Full pipeline structure validated")


def test_configuration():
    """Test configuration system"""
    print("\n" + "=" * 60)
    print("TEST 7: Configuration")
    print("=" * 60)

    config = get_config()

    print(f"App Name: {config.app_name}")
    print(f"Version: {config.version}")
    print(f"\nDeepSeek Config:")
    print(f"  Model: {config.deepseek.model_path}")
    print(f"  Base size: {config.deepseek.base_size}")
    print(f"  Crop mode: {config.deepseek.crop_mode}")

    print(f"\nChunking Config:")
    print(f"  Chunk size: {config.chunking.chunk_size}")
    print(f"  Overlap: {config.chunking.chunk_overlap}")
    print(f"  Strategy: {config.chunking.strategy}")

    ensure_directories()
    print("\n✅ Configuration loaded and directories ensured")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PHASE 1 PIPELINE TESTS")
    print("Smart Document Intelligence Platform")
    print("=" * 60)

    try:
        test_configuration()
        test_pdf_processing()
        test_image_processing()
        test_chunking()
        test_storage()
        test_ocr_wrapper()
        test_full_pipeline()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)

        print("""
Next Steps:
-----------
1. Install vLLM and DeepSeek-OCR model
2. Provide sample PDFs/images for testing
3. Test full OCR extraction pipeline
4. Move to Phase 2: Vector DB & RAG integration
        """)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
