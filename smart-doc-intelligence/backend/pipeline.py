"""
Main OCR Pipeline
Orchestrates the complete document processing workflow
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ocr.pdf_processor import PDFProcessor
from backend.ocr.image_processor import ImageProcessor
from backend.ocr.deepseek_wrapper import DeepSeekOCR
from backend.utils.chunking import DocumentChunker
from backend.utils.storage import DocumentStorage
from backend.utils.config import get_config


class DocumentPipeline:
    """
    Main pipeline for processing documents
    Handles: Upload → OCR → Chunking → Storage
    """

    def __init__(self, load_ocr_model: bool = False):
        """
        Initialize pipeline

        Args:
            load_ocr_model: Whether to load OCR model immediately
        """
        self.config = get_config()

        # Initialize components
        self.pdf_processor = PDFProcessor(dpi=144)
        self.image_processor = ImageProcessor()
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
        )
        self.storage = DocumentStorage()

        # OCR (loaded on-demand due to GPU requirements)
        self.ocr = None
        if load_ocr_model:
            self.load_ocr()

        print("✅ Document Pipeline initialized")

    def load_ocr(self, batch_mode: bool = False):
        """Load DeepSeek-OCR model"""
        print("🔄 Loading OCR model...")
        self.ocr = DeepSeekOCR()
        self.ocr.load_model(batch_mode=batch_mode)
        print("✅ OCR model loaded")

    def process_pdf(
        self,
        pdf_path: str,
        prompt_type: str = "document",
        save_to_storage: bool = True
    ) -> Dict[str, Any]:
        """
        Process a PDF document

        Args:
            pdf_path: Path to PDF file
            prompt_type: OCR prompt type ('document', 'free', 'figure', 'detail')
            save_to_storage: Whether to save to storage system

        Returns:
            Processing results dictionary
        """
        print(f"\n📄 Processing PDF: {Path(pdf_path).name}")

        # Step 1: Save to storage
        doc_id = None
        if save_to_storage:
            result = self.storage.save_upload(pdf_path, doc_type="pdf")
            if result.get("success"):
                doc_id = result["doc_id"]
                print(f"   Document ID: {doc_id}")

        # Step 2: Convert PDF to images
        print("   Converting PDF to images...")
        images = self.pdf_processor.pdf_to_images(pdf_path)

        if not images:
            return {"success": False, "error": "Failed to convert PDF"}

        # Step 3: Extract text with OCR
        if not self.ocr:
            return {
                "success": False,
                "error": "OCR model not loaded. Call load_ocr() first."
            }

        print(f"   Extracting text from {len(images)} pages...")
        all_text = []
        page_results = []

        for idx, image in enumerate(tqdm(images, desc="   OCR Progress")):
            # Save image temporarily
            temp_path = f"/tmp/page_{idx}.png"
            image.save(temp_path)

            # Extract text
            result = self.ocr.extract_text(temp_path, prompt_type=prompt_type)

            if result.get("success"):
                page_text = result["text"]
                all_text.append(f"# Page {idx + 1}\n\n{page_text}")
                page_results.append({
                    "page": idx + 1,
                    "text_length": len(page_text),
                    "success": True
                })
            else:
                page_results.append({
                    "page": idx + 1,
                    "error": result.get("error"),
                    "success": False
                })

        combined_text = "\n\n---\n\n".join(all_text)

        # Step 4: Chunk the text
        print(f"   Chunking text...")
        chunks = self.chunker.chunk_document(
            combined_text,
            strategy=self.config.chunking.strategy,
            metadata={"doc_id": doc_id, "source": "pdf"}
        )

        chunk_dicts = [chunk.to_dict() for chunk in chunks]

        # Step 5: Save processed results
        if save_to_storage and doc_id:
            print(f"   Saving processed results...")
            self.storage.save_processed(
                doc_id,
                extracted_text=combined_text,
                chunks=chunk_dicts,
                additional_data={
                    "page_count": len(images),
                    "page_results": page_results,
                    "prompt_type": prompt_type,
                }
            )

        print(f"✅ Processing complete!")
        print(f"   Total text: {len(combined_text)} characters")
        print(f"   Chunks: {len(chunks)}")

        return {
            "success": True,
            "doc_id": doc_id,
            "page_count": len(images),
            "text_length": len(combined_text),
            "chunk_count": len(chunks),
            "extracted_text": combined_text,
            "chunks": chunk_dicts,
            "page_results": page_results,
        }

    def process_image(
        self,
        image_path: str,
        prompt_type: str = "document",
        enhance: bool = True,
        save_to_storage: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single image

        Args:
            image_path: Path to image file
            prompt_type: OCR prompt type
            enhance: Whether to enhance image for OCR
            save_to_storage: Whether to save to storage

        Returns:
            Processing results dictionary
        """
        print(f"\n🖼️  Processing Image: {Path(image_path).name}")

        # Step 1: Save to storage
        doc_id = None
        if save_to_storage:
            result = self.storage.save_upload(image_path, doc_type="image")
            if result.get("success"):
                doc_id = result["doc_id"]
                print(f"   Document ID: {doc_id}")

        # Step 2: Load and enhance image
        print("   Loading image...")
        image = self.image_processor.load_image(image_path)

        if not image:
            return {"success": False, "error": "Failed to load image"}

        if enhance:
            print("   Enhancing image...")
            image = self.image_processor.enhance_for_ocr(image)

            # Save enhanced image temporarily
            enhanced_path = "/tmp/enhanced_image.png"
            self.image_processor.save_image(image, enhanced_path)
            process_path = enhanced_path
        else:
            process_path = image_path

        # Step 3: Extract text
        if not self.ocr:
            return {
                "success": False,
                "error": "OCR model not loaded. Call load_ocr() first."
            }

        print("   Extracting text...")
        result = self.ocr.extract_text(process_path, prompt_type=prompt_type)

        if not result.get("success"):
            return result

        extracted_text = result["text"]

        # Step 4: Chunk the text
        print("   Chunking text...")
        chunks = self.chunker.chunk_document(
            extracted_text,
            strategy=self.config.chunking.strategy,
            metadata={"doc_id": doc_id, "source": "image"}
        )

        chunk_dicts = [chunk.to_dict() for chunk in chunks]

        # Step 5: Save processed results
        if save_to_storage and doc_id:
            print("   Saving processed results...")
            self.storage.save_processed(
                doc_id,
                extracted_text=extracted_text,
                chunks=chunk_dicts,
                additional_data={
                    "prompt_type": prompt_type,
                    "enhanced": enhance,
                    "image_size": image.size,
                }
            )

        print(f"✅ Processing complete!")
        print(f"   Text length: {len(extracted_text)} characters")
        print(f"   Chunks: {len(chunks)}")

        return {
            "success": True,
            "doc_id": doc_id,
            "text_length": len(extracted_text),
            "chunk_count": len(chunks),
            "extracted_text": extracted_text,
            "chunks": chunk_dicts,
        }

    def process_batch(
        self,
        file_paths: List[str],
        prompt_type: str = "document"
    ) -> List[Dict[str, Any]]:
        """
        Process multiple files in batch

        Args:
            file_paths: List of file paths
            prompt_type: OCR prompt type

        Returns:
            List of processing results
        """
        print(f"\n📦 Batch Processing: {len(file_paths)} files")

        results = []

        for file_path in file_paths:
            path = Path(file_path)

            if path.suffix.lower() == '.pdf':
                result = self.process_pdf(file_path, prompt_type)
            elif path.suffix.lower() in self.image_processor.supported_formats:
                result = self.process_image(file_path, prompt_type)
            else:
                result = {
                    "success": False,
                    "error": f"Unsupported file format: {path.suffix}",
                    "file_path": file_path
                }

            results.append(result)

        # Summary
        success_count = sum(1 for r in results if r.get("success"))
        print(f"\n✅ Batch processing complete:")
        print(f"   Successful: {success_count}/{len(file_paths)}")

        return results

    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a processed document"""
        metadata = self.storage.get_metadata(doc_id)

        if not metadata:
            return None

        # Get processed text and chunks
        text = self.storage.get_processed_text(doc_id)
        chunks = self.storage.get_chunks(doc_id)

        return {
            **metadata,
            "has_text": text is not None,
            "has_chunks": chunks is not None,
            "text_preview": text[:200] if text else None,
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents"""
        return self.storage.list_documents()

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.storage.get_storage_stats()


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Document Processing Pipeline")
    parser.add_argument("--file", help="File to process")
    parser.add_argument("--type", default="document", help="Prompt type")
    parser.add_argument("--load-model", action="store_true", help="Load OCR model")
    parser.add_argument("--list", action="store_true", help="List documents")
    parser.add_argument("--stats", action="store_true", help="Show statistics")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = DocumentPipeline(load_ocr_model=args.load_model)

    if args.list:
        docs = pipeline.list_documents()
        print(f"\n📚 Documents ({len(docs)}):")
        for doc in docs:
            print(f"  - {doc['original_filename']} ({doc['status']})")

    elif args.stats:
        stats = pipeline.get_statistics()
        print("\n📊 Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.file:
        # Process file
        if not pipeline.ocr:
            print("❌ OCR model not loaded. Use --load-model flag.")
        else:
            result = pipeline.process_pdf(args.file, prompt_type=args.type) \
                if args.file.endswith('.pdf') \
                else pipeline.process_image(args.file, prompt_type=args.type)

            if result.get("success"):
                print(f"\n✅ Success! Document ID: {result.get('doc_id')}")
            else:
                print(f"\n❌ Failed: {result.get('error')}")

    else:
        print("Usage: python pipeline.py --help")
