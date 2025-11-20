"""
Document Retrieval Module
Handles semantic search and document retrieval for RAG
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from backend.vectordb.chroma_manager import ChromaManager
from backend.vectordb.embeddings import EmbeddingGenerator
from backend.utils.config import get_config


@dataclass
class RetrievalResult:
    """Represents a single retrieval result"""
    doc_id: str
    chunk_id: int
    text: str
    score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


class DocumentRetriever:
    """
    Handles semantic search and retrieval of document chunks
    """

    def __init__(
        self,
        collection_name: str = "documents",
        chroma_manager: Optional[ChromaManager] = None
    ):
        """
        Initialize document retriever

        Args:
            collection_name: Name of ChromaDB collection
            chroma_manager: ChromaDB manager instance
        """
        self.collection_name = collection_name
        self.config = get_config().chromadb

        # Initialize ChromaDB manager
        self.chroma = chroma_manager or ChromaManager()

        # Ensure collection exists
        self.collection = self.chroma.get_or_create_collection(collection_name)

        print(f"✅ Document retriever initialized")
        print(f"   Collection: {collection_name}")

    def index_document(
        self,
        doc_id: str,
        chunks: List[Dict[str, Any]]
    ) -> bool:
        """
        Index a document's chunks into the vector database

        Args:
            doc_id: Document ID
            chunks: List of chunk dictionaries from DocumentChunker

        Returns:
            True if successful
        """
        print(f"📊 Indexing document: {doc_id}")
        print(f"   Chunks: {len(chunks)}")

        success = self.chroma.add_chunks(
            collection_name=self.collection_name,
            chunks=chunks,
            doc_id=doc_id
        )

        if success:
            print(f"✅ Document indexed successfully")

        return success

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_doc_id: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Semantic search across all documents

        Args:
            query: Search query text
            top_k: Number of results to return
            filter_doc_id: Optional filter by specific document ID
            score_threshold: Minimum similarity score threshold

        Returns:
            List of retrieval results
        """
        # Build where filter
        where_filter = {"doc_id": filter_doc_id} if filter_doc_id else None

        # Search ChromaDB
        results = self.chroma.search(
            collection_name=self.collection_name,
            query=query,
            n_results=top_k,
            where=where_filter
        )

        # Parse results
        retrieval_results = self._parse_chroma_results(results, score_threshold)

        return retrieval_results

    def search_by_document(
        self,
        query: str,
        doc_id: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Search within a specific document

        Args:
            query: Search query
            doc_id: Document ID to search within
            top_k: Number of results

        Returns:
            List of retrieval results
        """
        return self.search(
            query=query,
            top_k=top_k,
            filter_doc_id=doc_id
        )

    def get_context_for_query(
        self,
        query: str,
        top_k: int = 5,
        max_context_length: int = 2000
    ) -> str:
        """
        Get relevant context for a query (for RAG)

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            max_context_length: Maximum total context length

        Returns:
            Combined context string
        """
        # Retrieve relevant chunks
        results = self.search(query, top_k=top_k)

        # Combine into context
        context_parts = []
        total_length = 0

        for result in results:
            chunk_text = result.text
            chunk_length = len(chunk_text)

            if total_length + chunk_length > max_context_length:
                break

            context_parts.append(f"[Source: {result.doc_id}, Chunk {result.chunk_id}]")
            context_parts.append(chunk_text)
            context_parts.append("")  # Empty line for separation

            total_length += chunk_length

        context = "\n".join(context_parts)

        print(f"📝 Retrieved context:")
        print(f"   Chunks: {len(results)}")
        print(f"   Total length: {total_length} characters")

        return context

    def get_document_chunks(
        self,
        doc_id: str,
        limit: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Get all chunks for a document

        Args:
            doc_id: Document ID
            limit: Optional limit on number of chunks

        Returns:
            List of retrieval results
        """
        # This is a dummy query to get all chunks with the doc_id filter
        # ChromaDB doesn't have a direct "get all" with filter, so we use search
        results = self.chroma.search(
            collection_name=self.collection_name,
            query="",  # Empty query
            n_results=limit or 100,  # Get many results
            where={"doc_id": doc_id}
        )

        return self._parse_chroma_results(results)

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete all chunks for a document from the index

        Args:
            doc_id: Document ID

        Returns:
            True if successful
        """
        return self.chroma.delete_document_chunks(
            collection_name=self.collection_name,
            doc_id=doc_id
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get retrieval statistics

        Returns:
            Dictionary with statistics
        """
        stats = self.chroma.get_collection_stats(self.collection_name)
        return stats

    def _parse_chroma_results(
        self,
        results: Dict[str, Any],
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Parse ChromaDB results into RetrievalResult objects

        Args:
            results: Raw results from ChromaDB
            score_threshold: Optional score threshold

        Returns:
            List of RetrievalResult objects
        """
        retrieval_results = []

        # ChromaDB returns results as nested lists
        ids = results.get('ids', [[]])[0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        for i in range(len(ids)):
            # Convert distance to similarity score (lower distance = higher similarity)
            # ChromaDB uses L2 distance, so we convert it
            distance = distances[i] if distances else 0.0
            score = 1.0 / (1.0 + distance)  # Convert to similarity

            # Apply threshold if provided
            if score_threshold and score < score_threshold:
                continue

            metadata = metadatas[i] if metadatas else {}

            result = RetrievalResult(
                doc_id=metadata.get('doc_id', 'unknown'),
                chunk_id=int(metadata.get('chunk_id', 0)),
                text=documents[i],
                score=score,
                metadata=metadata
            )

            retrieval_results.append(result)

        return retrieval_results


class MultiDocumentRetriever:
    """
    Advanced retriever that handles multiple documents and collections
    """

    def __init__(self, chroma_manager: Optional[ChromaManager] = None):
        """
        Initialize multi-document retriever

        Args:
            chroma_manager: ChromaDB manager instance
        """
        self.chroma = chroma_manager or ChromaManager()
        self.retrievers = {}  # Cache of retrievers per collection

        print(f"✅ Multi-document retriever initialized")

    def get_retriever(self, collection_name: str) -> DocumentRetriever:
        """
        Get or create a retriever for a collection

        Args:
            collection_name: Collection name

        Returns:
            DocumentRetriever instance
        """
        if collection_name not in self.retrievers:
            self.retrievers[collection_name] = DocumentRetriever(
                collection_name=collection_name,
                chroma_manager=self.chroma
            )

        return self.retrievers[collection_name]

    def search_across_collections(
        self,
        query: str,
        collection_names: List[str],
        top_k_per_collection: int = 3
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Search across multiple collections

        Args:
            query: Search query
            collection_names: List of collection names to search
            top_k_per_collection: Number of results per collection

        Returns:
            Dictionary mapping collection names to results
        """
        all_results = {}

        for collection_name in collection_names:
            retriever = self.get_retriever(collection_name)
            results = retriever.search(query, top_k=top_k_per_collection)
            all_results[collection_name] = results

        return all_results

    def merge_results(
        self,
        results_dict: Dict[str, List[RetrievalResult]],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Merge and rank results from multiple collections

        Args:
            results_dict: Results from multiple collections
            top_k: Number of top results to return

        Returns:
            Merged and sorted list of results
        """
        all_results = []

        for collection_name, results in results_dict.items():
            all_results.extend(results)

        # Sort by score (descending)
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[:top_k]


# Convenience functions
def create_retriever(collection_name: str = "documents") -> DocumentRetriever:
    """Create document retriever"""
    return DocumentRetriever(collection_name=collection_name)


def quick_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Quick search function

    Args:
        query: Search query
        top_k: Number of results

    Returns:
        List of result dictionaries
    """
    retriever = DocumentRetriever()
    results = retriever.search(query, top_k=top_k)
    return [r.to_dict() for r in results]


if __name__ == "__main__":
    # Test document retriever
    print("Document Retriever Test")
    print("=" * 50)

    # Initialize
    retriever = DocumentRetriever(collection_name="test_documents")

    # Index sample chunks
    sample_chunks = [
        {
            "text": "Machine learning is a method of data analysis.",
            "chunk_id": 0,
            "start_char": 0,
            "end_char": 50,
            "metadata": {"topic": "ml"}
        },
        {
            "text": "Deep learning uses neural networks with many layers.",
            "chunk_id": 1,
            "start_char": 50,
            "end_char": 100,
            "metadata": {"topic": "dl"}
        },
    ]

    retriever.index_document("test_doc_1", sample_chunks)

    # Search
    results = retriever.search("neural networks", top_k=2)

    print(f"\nSearch results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result.score:.4f}")
        print(f"     Doc: {result.doc_id}, Chunk: {result.chunk_id}")
        print(f"     Text: {result.text[:80]}...")

    # Get context
    context = retriever.get_context_for_query("machine learning", top_k=2)
    print(f"\nContext preview:")
    print(context[:200])

    # Statistics
    stats = retriever.get_statistics()
    print(f"\nStatistics: {stats}")
