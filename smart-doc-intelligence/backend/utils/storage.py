"""
File Storage System
Manages document uploads, processing results, and metadata
"""
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import hashlib


class DocumentStorage:
    """
    Manages file storage for uploaded and processed documents
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize document storage

        Args:
            base_dir: Base directory for storage (uses config default if None)
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            from backend.utils.config import get_config
            config = get_config()
            self.base_dir = Path(config.storage.uploads_dir).parent

        self.uploads_dir = self.base_dir / "uploads"
        self.processed_dir = self.base_dir / "processed"
        self.metadata_dir = self.base_dir / "metadata"

        # Create directories
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories"""
        for directory in [self.uploads_dir, self.processed_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _generate_doc_id(self, filename: str) -> str:
        """
        Generate unique document ID

        Args:
            filename: Original filename

        Returns:
            Unique document ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        return f"{timestamp}_{file_hash}"

    def save_upload(
        self,
        file_path: str,
        doc_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save uploaded file to storage

        Args:
            file_path: Path to file to save
            doc_type: Type of document (pdf, image, etc.)
            metadata: Additional metadata

        Returns:
            Dictionary with document information
        """
        try:
            source_path = Path(file_path)

            if not source_path.exists():
                return {"success": False, "error": "File not found"}

            # Generate document ID
            doc_id = self._generate_doc_id(source_path.name)

            # Create document directory
            doc_dir = self.uploads_dir / doc_id
            doc_dir.mkdir(exist_ok=True)

            # Copy file
            dest_path = doc_dir / source_path.name
            shutil.copy2(source_path, dest_path)

            # Create metadata
            doc_metadata = {
                "doc_id": doc_id,
                "original_filename": source_path.name,
                "file_path": str(dest_path),
                "doc_type": doc_type,
                "file_size": source_path.stat().st_size,
                "upload_time": datetime.now().isoformat(),
                "status": "uploaded",
                "processed": False,
                "custom_metadata": metadata or {},
            }

            # Save metadata
            self._save_metadata(doc_id, doc_metadata)

            print(f"✅ Saved upload: {doc_id}")
            return {"success": True, "doc_id": doc_id, **doc_metadata}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_processed(
        self,
        doc_id: str,
        extracted_text: str,
        chunks: Optional[List[Dict[str, Any]]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save processed document results

        Args:
            doc_id: Document ID
            extracted_text: Extracted text from OCR
            chunks: List of text chunks
            additional_data: Any additional processing data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create processed directory for this document
            proc_dir = self.processed_dir / doc_id
            proc_dir.mkdir(exist_ok=True)

            # Save extracted text
            text_file = proc_dir / "extracted_text.md"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)

            # Save chunks if provided
            if chunks:
                chunks_file = proc_dir / "chunks.json"
                with open(chunks_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, indent=2, ensure_ascii=False)

            # Save additional data
            if additional_data:
                data_file = proc_dir / "processing_data.json"
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(additional_data, f, indent=2, ensure_ascii=False)

            # Update metadata
            metadata = self.get_metadata(doc_id)
            if metadata:
                metadata["processed"] = True
                metadata["status"] = "processed"
                metadata["processing_time"] = datetime.now().isoformat()
                metadata["text_length"] = len(extracted_text)
                metadata["chunk_count"] = len(chunks) if chunks else 0

                self._save_metadata(doc_id, metadata)

            print(f"✅ Saved processed results for: {doc_id}")
            return True

        except Exception as e:
            print(f"❌ Error saving processed results: {e}")
            return False

    def _save_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        """Save metadata to JSON file"""
        metadata_file = self.metadata_dir / f"{doc_id}.json"

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a document

        Args:
            doc_id: Document ID

        Returns:
            Metadata dictionary or None if not found
        """
        metadata_file = self.metadata_dir / f"{doc_id}.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            return None

    def get_document_path(self, doc_id: str) -> Optional[str]:
        """Get path to original uploaded document"""
        metadata = self.get_metadata(doc_id)
        if metadata:
            return metadata.get("file_path")
        return None

    def get_processed_text(self, doc_id: str) -> Optional[str]:
        """
        Get processed text for a document

        Args:
            doc_id: Document ID

        Returns:
            Extracted text or None if not found
        """
        text_file = self.processed_dir / doc_id / "extracted_text.md"

        if not text_file.exists():
            return None

        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Error loading processed text: {e}")
            return None

    def get_chunks(self, doc_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get chunks for a document

        Args:
            doc_id: Document ID

        Returns:
            List of chunks or None if not found
        """
        chunks_file = self.processed_dir / doc_id / "chunks.json"

        if not chunks_file.exists():
            return None

        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading chunks: {e}")
            return None

    def list_documents(
        self,
        status: Optional[str] = None,
        doc_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all documents with optional filters

        Args:
            status: Filter by status ('uploaded', 'processed', etc.)
            doc_type: Filter by document type

        Returns:
            List of document metadata
        """
        documents = []

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                    # Apply filters
                    if status and metadata.get("status") != status:
                        continue

                    if doc_type and metadata.get("doc_type") != doc_type:
                        continue

                    documents.append(metadata)

            except Exception as e:
                print(f"⚠️ Error reading {metadata_file}: {e}")

        # Sort by upload time (newest first)
        documents.sort(key=lambda x: x.get("upload_time", ""), reverse=True)

        return documents

    def delete_document(self, doc_id: str, delete_processed: bool = True) -> bool:
        """
        Delete a document and optionally its processed results

        Args:
            doc_id: Document ID
            delete_processed: Whether to delete processed results too

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete uploaded file
            upload_dir = self.uploads_dir / doc_id
            if upload_dir.exists():
                shutil.rmtree(upload_dir)

            # Delete processed files
            if delete_processed:
                proc_dir = self.processed_dir / doc_id
                if proc_dir.exists():
                    shutil.rmtree(proc_dir)

            # Delete metadata
            metadata_file = self.metadata_dir / f"{doc_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()

            print(f"✅ Deleted document: {doc_id}")
            return True

        except Exception as e:
            print(f"❌ Error deleting document: {e}")
            return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics

        Returns:
            Dictionary with storage statistics
        """
        documents = self.list_documents()

        total_size = 0
        processed_count = 0

        for doc in documents:
            total_size += doc.get("file_size", 0)
            if doc.get("processed", False):
                processed_count += 1

        return {
            "total_documents": len(documents),
            "processed_documents": processed_count,
            "pending_documents": len(documents) - processed_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "uploads_dir": str(self.uploads_dir),
            "processed_dir": str(self.processed_dir),
        }

    def search_documents(
        self,
        query: str,
        search_in: str = "filename"
    ) -> List[Dict[str, Any]]:
        """
        Search documents by filename or metadata

        Args:
            query: Search query
            search_in: Where to search ('filename', 'metadata')

        Returns:
            List of matching documents
        """
        documents = self.list_documents()
        results = []

        query_lower = query.lower()

        for doc in documents:
            if search_in == "filename":
                if query_lower in doc.get("original_filename", "").lower():
                    results.append(doc)
            elif search_in == "metadata":
                metadata_str = json.dumps(doc).lower()
                if query_lower in metadata_str:
                    results.append(doc)

        return results

    def export_document_list(self, output_path: str) -> bool:
        """
        Export document list to JSON file

        Args:
            output_path: Path to output file

        Returns:
            True if successful, False otherwise
        """
        try:
            documents = self.list_documents()

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)

            print(f"✅ Exported document list to: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Error exporting document list: {e}")
            return False


# Convenience functions
def save_document(file_path: str, doc_type: str = "unknown") -> Optional[str]:
    """
    Quick function to save a document

    Args:
        file_path: Path to file
        doc_type: Document type

    Returns:
        Document ID or None if failed
    """
    storage = DocumentStorage()
    result = storage.save_upload(file_path, doc_type)

    if result.get("success"):
        return result.get("doc_id")
    return None


def get_document_text(doc_id: str) -> Optional[str]:
    """Quick function to get processed text"""
    storage = DocumentStorage()
    return storage.get_processed_text(doc_id)


if __name__ == "__main__":
    # Test storage system
    print("Document Storage Test")
    print("=" * 50)

    storage = DocumentStorage()
    stats = storage.get_storage_stats()

    print("\nStorage Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nDocuments:")
    docs = storage.list_documents()
    for doc in docs[:5]:  # Show first 5
        print(f"  - {doc['original_filename']} ({doc['status']})")
