"""
Embedding Generation Module
Handles text embedding creation for semantic search
"""
from typing import List, Optional, Dict, Any
import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("⚠️ sentence-transformers not installed. Run: pip install sentence-transformers")

from backend.utils.config import get_config


class EmbeddingGenerator:
    """
    Generates embeddings for text using sentence transformers
    """

    def __init__(self, model_name: Optional[str] = None, device: str = "cuda"):
        """
        Initialize embedding generator

        Args:
            model_name: Name of sentence transformer model
            device: Device to use ('cuda' or 'cpu')
        """
        config = get_config().chromadb

        self.model_name = model_name or config.embedding_model
        self.device = device
        self.dimension = config.embedding_dimension

        # Load model
        print(f"🔄 Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=device)

        # Get actual dimension from model
        self.dimension = self.model.get_sentence_embedding_dimension()

        print(f"✅ Embedding model loaded")
        print(f"   Model: {self.model_name}")
        print(f"   Dimension: {self.dimension}")
        print(f"   Device: {device}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Generate embeddings for document chunks

        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for encoding

        Returns:
            List of embedding vectors
        """
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]

        print(f"🔄 Generating embeddings for {len(texts)} chunks...")

        # Generate embeddings
        embeddings = self.embed_batch(texts, batch_size=batch_size)

        print(f"✅ Generated {len(embeddings)} embeddings")

        return list(embeddings)

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine' or 'euclidean')

        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
        elif metric == "euclidean":
            # Euclidean distance (lower is more similar)
            similarity = -np.linalg.norm(embedding1 - embedding2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return float(similarity)

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar embeddings to query

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []

        for idx, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((idx, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
        }


class HybridEmbedding:
    """
    Combines multiple embedding approaches for better retrieval
    """

    def __init__(
        self,
        semantic_model: Optional[str] = None,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7
    ):
        """
        Initialize hybrid embedding

        Args:
            semantic_model: Semantic embedding model name
            keyword_weight: Weight for keyword matching
            semantic_weight: Weight for semantic similarity
        """
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

        # Semantic embedding generator
        self.semantic_generator = EmbeddingGenerator(model_name=semantic_model)

        print(f"✅ Hybrid embedding initialized")
        print(f"   Keyword weight: {keyword_weight}")
        print(f"   Semantic weight: {semantic_weight}")

    def compute_keyword_similarity(
        self,
        query: str,
        document: str
    ) -> float:
        """
        Compute keyword-based similarity (BM25-like)

        Args:
            query: Query text
            document: Document text

        Returns:
            Keyword similarity score
        """
        # Simple keyword overlap (can be improved with BM25)
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words & doc_words)
        similarity = overlap / len(query_words)

        return similarity

    def compute_hybrid_score(
        self,
        query: str,
        document: str,
        query_embedding: Optional[np.ndarray] = None,
        doc_embedding: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute hybrid similarity score

        Args:
            query: Query text
            document: Document text
            query_embedding: Pre-computed query embedding
            doc_embedding: Pre-computed document embedding

        Returns:
            Hybrid similarity score
        """
        # Keyword similarity
        keyword_sim = self.compute_keyword_similarity(query, document)

        # Semantic similarity
        if query_embedding is None:
            query_embedding = self.semantic_generator.embed_text(query)

        if doc_embedding is None:
            doc_embedding = self.semantic_generator.embed_text(document)

        semantic_sim = self.semantic_generator.compute_similarity(
            query_embedding,
            doc_embedding
        )

        # Combine scores
        hybrid_score = (
            self.keyword_weight * keyword_sim +
            self.semantic_weight * semantic_sim
        )

        return hybrid_score


# Convenience functions
def create_embedding_generator(
    model_name: Optional[str] = None,
    device: str = "cuda"
) -> EmbeddingGenerator:
    """
    Create embedding generator

    Args:
        model_name: Model name (uses config default if None)
        device: Device to use

    Returns:
        EmbeddingGenerator instance
    """
    return EmbeddingGenerator(model_name=model_name, device=device)


def embed_single_text(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Quick function to embed a single text

    Args:
        text: Input text
        model_name: Model name

    Returns:
        Embedding vector
    """
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.embed_text(text)


def embed_documents(
    documents: List[str],
    model_name: Optional[str] = None,
    batch_size: int = 32
) -> np.ndarray:
    """
    Quick function to embed multiple documents

    Args:
        documents: List of documents
        model_name: Model name
        batch_size: Batch size

    Returns:
        Array of embeddings
    """
    generator = EmbeddingGenerator(model_name=model_name)
    return generator.embed_batch(documents, batch_size=batch_size)


if __name__ == "__main__":
    # Test embedding generator
    print("Embedding Generator Test")
    print("=" * 50)

    # Initialize
    generator = EmbeddingGenerator(device="cpu")  # Use CPU for testing

    # Test single embedding
    text = "This is a test document about machine learning."
    embedding = generator.embed_text(text)

    print(f"\nSingle text embedding:")
    print(f"  Text: {text}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding (first 5): {embedding[:5]}")

    # Test batch embedding
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing deals with text data.",
    ]

    embeddings = generator.embed_batch(texts, batch_size=2, show_progress=False)

    print(f"\nBatch embedding:")
    print(f"  Number of texts: {len(texts)}")
    print(f"  Embeddings shape: {embeddings.shape}")

    # Test similarity
    sim = generator.compute_similarity(embeddings[0], embeddings[1])
    print(f"\nSimilarity between first two texts: {sim:.4f}")

    # Model info
    info = generator.get_model_info()
    print(f"\nModel info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
