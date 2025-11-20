"""
ChromaDB Manager
Handles vector database operations for document storage and retrieval
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    print("⚠️ ChromaDB not installed. Run: pip install chromadb")

from backend.utils.config import get_config


class ChromaManager:
    """
    Manages ChromaDB collections and operations
    Handles document indexing, search, and retrieval
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB manager

        Args:
            persist_directory: Directory for persistent storage
        """
        config = get_config().chromadb

        self.persist_directory = persist_directory or config.persist_directory
        self.embedding_model_name = config.embedding_model
        self.default_collection = config.collection_name

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )

        print(f"✅ ChromaDB initialized")
        print(f"   Persist directory: {self.persist_directory}")
        print(f"   Embedding model: {self.embedding_model_name}")

    def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> chromadb.Collection:
        """
        Create a new collection

        Args:
            collection_name: Name of the collection
            metadata: Optional metadata for the collection

        Returns:
            ChromaDB collection object
        """
        try:
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata=metadata or {}
            )
            print(f"✅ Created collection: {collection_name}")
            return collection

        except Exception as e:
            print(f"⚠️ Collection may already exist: {e}")
            return self.get_collection(collection_name)

    def get_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Get an existing collection

        Args:
            collection_name: Name of the collection

        Returns:
            ChromaDB collection object
        """
        try:
            return self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print(f"❌ Error getting collection: {e}")
            raise

    def get_or_create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> chromadb.Collection:
        """
        Get existing collection or create if doesn't exist

        Args:
            collection_name: Name of the collection
            metadata: Optional metadata

        Returns:
            ChromaDB collection object
        """
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata=metadata or {}
        )

    def list_collections(self) -> List[str]:
        """
        List all collections

        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [col.name for col in collections]

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection

        Args:
            collection_name: Name of collection to delete

        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=collection_name)
            print(f"✅ Deleted collection: {collection_name}")
            return True
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
            return False

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents to a collection

        Args:
            collection_name: Name of the collection
            documents: List of document texts
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs

        Returns:
            True if successful
        """
        try:
            collection = self.get_or_create_collection(collection_name)

            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(documents))]

            # Add to collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            print(f"✅ Added {len(documents)} documents to {collection_name}")
            return True

        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            return False

    def add_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        doc_id: str
    ) -> bool:
        """
        Add document chunks to collection

        Args:
            collection_name: Name of the collection
            chunks: List of chunk dictionaries from DocumentChunker
            doc_id: Document ID

        Returns:
            True if successful
        """
        try:
            # Extract text and metadata
            documents = [chunk['text'] for chunk in chunks]

            metadatas = []
            ids = []

            for chunk in chunks:
                # Create metadata
                metadata = {
                    "doc_id": doc_id,
                    "chunk_id": chunk['chunk_id'],
                    "start_char": chunk['start_char'],
                    "end_char": chunk['end_char'],
                    **chunk.get('metadata', {})
                }

                # Convert any non-string values to strings for ChromaDB
                metadata = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                           for k, v in metadata.items()}

                metadatas.append(metadata)

                # Create unique ID
                chunk_id = f"{doc_id}_chunk_{chunk['chunk_id']}"
                ids.append(chunk_id)

            # Add to collection
            return self.add_documents(
                collection_name=collection_name,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        except Exception as e:
            print(f"❌ Error adding chunks: {e}")
            return False

    def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Semantic search in collection

        Args:
            collection_name: Name of the collection
            query: Search query text
            n_results: Number of results to return
            where: Metadata filter (e.g., {"doc_id": "123"})
            where_document: Document content filter

        Returns:
            Search results dictionary
        """
        try:
            collection = self.get_collection(collection_name)

            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                where_document=where_document
            )

            return results

        except Exception as e:
            print(f"❌ Search error: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def search_by_document(
        self,
        collection_name: str,
        query: str,
        doc_id: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search within a specific document

        Args:
            collection_name: Name of the collection
            query: Search query
            doc_id: Document ID to search within
            n_results: Number of results

        Returns:
            Search results
        """
        return self.search(
            collection_name=collection_name,
            query=query,
            n_results=n_results,
            where={"doc_id": doc_id}
        )

    def get_by_id(
        self,
        collection_name: str,
        ids: List[str]
    ) -> Dict[str, Any]:
        """
        Get documents by their IDs

        Args:
            collection_name: Name of the collection
            ids: List of document IDs

        Returns:
            Documents data
        """
        try:
            collection = self.get_collection(collection_name)
            return collection.get(ids=ids)
        except Exception as e:
            print(f"❌ Error getting documents: {e}")
            return {"ids": [], "documents": [], "metadatas": []}

    def delete_documents(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Delete documents from collection

        Args:
            collection_name: Name of the collection
            ids: List of document IDs to delete
            where: Metadata filter for deletion

        Returns:
            True if successful
        """
        try:
            collection = self.get_collection(collection_name)

            if ids:
                collection.delete(ids=ids)
            elif where:
                collection.delete(where=where)
            else:
                print("⚠️ No IDs or filter provided")
                return False

            print(f"✅ Deleted documents from {collection_name}")
            return True

        except Exception as e:
            print(f"❌ Error deleting documents: {e}")
            return False

    def delete_document_chunks(
        self,
        collection_name: str,
        doc_id: str
    ) -> bool:
        """
        Delete all chunks for a document

        Args:
            collection_name: Name of the collection
            doc_id: Document ID

        Returns:
            True if successful
        """
        return self.delete_documents(
            collection_name=collection_name,
            where={"doc_id": doc_id}
        )

    def count_documents(self, collection_name: str) -> int:
        """
        Count documents in collection

        Args:
            collection_name: Name of the collection

        Returns:
            Number of documents
        """
        try:
            collection = self.get_collection(collection_name)
            return collection.count()
        except Exception as e:
            print(f"❌ Error counting documents: {e}")
            return 0

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with statistics
        """
        try:
            collection = self.get_collection(collection_name)

            count = collection.count()

            # Get unique doc_ids
            if count > 0:
                all_data = collection.get()
                doc_ids = set()
                if all_data.get('metadatas'):
                    for metadata in all_data['metadatas']:
                        if metadata and 'doc_id' in metadata:
                            doc_ids.add(metadata['doc_id'])

                unique_docs = len(doc_ids)
            else:
                unique_docs = 0

            return {
                "collection_name": collection_name,
                "total_chunks": count,
                "unique_documents": unique_docs,
                "embedding_model": self.embedding_model_name,
            }

        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {"error": str(e)}

    def reset_database(self) -> bool:
        """
        Reset entire database (WARNING: Deletes all data)

        Returns:
            True if successful
        """
        try:
            self.client.reset()
            print("⚠️ Database reset complete")
            return True
        except Exception as e:
            print(f"❌ Error resetting database: {e}")
            return False


# Convenience functions
def create_chroma_manager(persist_directory: Optional[str] = None) -> ChromaManager:
    """Create and return ChromaDB manager"""
    return ChromaManager(persist_directory=persist_directory)


if __name__ == "__main__":
    # Test ChromaDB manager
    print("ChromaDB Manager Test")
    print("=" * 50)

    manager = ChromaManager()

    # List collections
    collections = manager.list_collections()
    print(f"\nCollections: {collections}")

    # Create test collection
    test_col = "test_documents"
    manager.get_or_create_collection(test_col)

    # Add sample documents
    docs = [
        "This is a test document about machine learning.",
        "Another document discussing artificial intelligence.",
        "A third document on natural language processing.",
    ]

    metadatas = [
        {"doc_id": "test_1", "topic": "ml"},
        {"doc_id": "test_2", "topic": "ai"},
        {"doc_id": "test_3", "topic": "nlp"},
    ]

    manager.add_documents(test_col, docs, metadatas)

    # Search
    results = manager.search(test_col, "machine learning models", n_results=2)
    print(f"\nSearch results:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"  {i+1}. {doc[:80]}...")

    # Stats
    stats = manager.get_collection_stats(test_col)
    print(f"\nCollection stats: {stats}")
