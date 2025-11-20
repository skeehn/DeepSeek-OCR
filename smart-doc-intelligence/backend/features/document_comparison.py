"""
Document Comparison Engine
Compares multiple documents for similarity, differences, and semantic overlap
"""
import re
import difflib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter

from backend.vectordb.embeddings import EmbeddingManager
from backend.llm.query_router import DualLLMManager, LLMType


@dataclass
class ComparisonResult:
    """Represents a comparison between documents"""
    doc_ids: List[str]
    similarity_scores: Dict[Tuple[str, str], float]
    common_terms: List[Tuple[str, int]]
    unique_terms: Dict[str, List[str]]
    semantic_similarity: Dict[Tuple[str, str], float]
    summary: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "doc_ids": self.doc_ids,
            "similarity_scores": {
                f"{k[0]}_vs_{k[1]}": v
                for k, v in self.similarity_scores.items()
            },
            "common_terms": [
                {"term": term, "count": count}
                for term, count in self.common_terms[:20]
            ],
            "unique_terms": self.unique_terms,
            "semantic_similarity": {
                f"{k[0]}_vs_{k[1]}": v
                for k, v in self.semantic_similarity.items()
            },
            "summary": self.summary,
            "metadata": self.metadata,
        }


@dataclass
class TextDiff:
    """Represents differences between two texts"""
    doc1_id: str
    doc2_id: str
    additions: List[str]
    deletions: List[str]
    modifications: List[Tuple[str, str]]
    unchanged_ratio: float
    diff_html: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "doc1_id": self.doc1_id,
            "doc2_id": self.doc2_id,
            "additions": self.additions[:10],  # First 10
            "deletions": self.deletions[:10],
            "modifications": [
                {"before": before, "after": after}
                for before, after in self.modifications[:10]
            ],
            "unchanged_ratio": self.unchanged_ratio,
            "diff_html": self.diff_html[:1000] + "..." if len(self.diff_html) > 1000 else self.diff_html,
        }


class DocumentComparator:
    """
    Compares documents using multiple strategies:
    - Lexical similarity (word overlap)
    - Semantic similarity (embedding distance)
    - Structural similarity (layout/format)
    - Diff analysis (line-by-line changes)
    """

    def __init__(self, use_embeddings: bool = True, use_llm: bool = True):
        """
        Initialize document comparator

        Args:
            use_embeddings: Use semantic embeddings
            use_llm: Use LLM for comparison summaries
        """
        self.use_embeddings = use_embeddings
        self.use_llm = use_llm

        self.embedding_manager = None
        self.llm_manager = None

        if use_embeddings:
            try:
                self.embedding_manager = EmbeddingManager()
            except Exception as e:
                print(f"⚠️ Could not initialize embeddings: {e}")
                self.use_embeddings = False

        if use_llm:
            try:
                self.llm_manager = DualLLMManager(prefer_local=True)
            except Exception as e:
                print(f"⚠️ Could not initialize LLM: {e}")
                self.use_llm = False

        # Stop words for filtering
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        print(f"✅ Document Comparator initialized")
        print(f"   Embeddings: {self.use_embeddings}")
        print(f"   LLM: {self.use_llm}")

    def compare_documents(
        self,
        documents: Dict[str, str],
        include_llm_summary: bool = True
    ) -> ComparisonResult:
        """
        Compare multiple documents

        Args:
            documents: Dictionary mapping doc_id to text
            include_llm_summary: Generate LLM summary

        Returns:
            ComparisonResult object
        """
        print(f"\n📊 Comparing {len(documents)} documents...")

        doc_ids = list(documents.keys())

        # Calculate pairwise similarities
        similarity_scores = {}
        semantic_similarity = {}

        for i, doc1_id in enumerate(doc_ids):
            for doc2_id in doc_ids[i + 1:]:
                # Lexical similarity
                lex_sim = self._calculate_lexical_similarity(
                    documents[doc1_id],
                    documents[doc2_id]
                )
                similarity_scores[(doc1_id, doc2_id)] = lex_sim

                # Semantic similarity
                if self.use_embeddings and self.embedding_manager:
                    sem_sim = self._calculate_semantic_similarity(
                        documents[doc1_id],
                        documents[doc2_id]
                    )
                    semantic_similarity[(doc1_id, doc2_id)] = sem_sim

        # Find common and unique terms
        common_terms = self._find_common_terms(documents)
        unique_terms = self._find_unique_terms(documents)

        # Generate LLM summary
        summary = ""
        if include_llm_summary and self.use_llm and self.llm_manager:
            summary = self._generate_comparison_summary(
                documents,
                similarity_scores,
                common_terms
            )

        # Build metadata
        metadata = {
            "num_documents": len(documents),
            "avg_similarity": sum(similarity_scores.values()) / len(similarity_scores) if similarity_scores else 0.0,
            "num_common_terms": len(common_terms),
        }

        result = ComparisonResult(
            doc_ids=doc_ids,
            similarity_scores=similarity_scores,
            common_terms=common_terms,
            unique_terms=unique_terms,
            semantic_similarity=semantic_similarity,
            summary=summary,
            metadata=metadata
        )

        print(f"✅ Comparison complete")
        print(f"   Average similarity: {metadata['avg_similarity']:.2%}")
        print(f"   Common terms: {len(common_terms)}")

        return result

    def diff_documents(
        self,
        doc1_id: str,
        doc1_text: str,
        doc2_id: str,
        doc2_text: str
    ) -> TextDiff:
        """
        Generate detailed diff between two documents

        Args:
            doc1_id: First document ID
            doc1_text: First document text
            doc2_id: Second document ID
            doc2_text: Second document text

        Returns:
            TextDiff object
        """
        print(f"\n🔍 Generating diff: {doc1_id} vs {doc2_id}")

        # Split into lines
        lines1 = doc1_text.split('\n')
        lines2 = doc2_text.split('\n')

        # Generate diff
        differ = difflib.Differ()
        diff = list(differ.compare(lines1, lines2))

        # Parse diff
        additions = []
        deletions = []
        modifications = []

        i = 0
        while i < len(diff):
            line = diff[i]

            if line.startswith('+ '):
                additions.append(line[2:])
            elif line.startswith('- '):
                # Check if next line is an addition (modification)
                if i + 1 < len(diff) and diff[i + 1].startswith('+ '):
                    modifications.append((line[2:], diff[i + 1][2:]))
                    i += 1  # Skip next line
                else:
                    deletions.append(line[2:])

            i += 1

        # Calculate unchanged ratio
        unchanged_lines = len([line for line in diff if line.startswith('  ')])
        total_lines = max(len(lines1), len(lines2))
        unchanged_ratio = unchanged_lines / total_lines if total_lines > 0 else 0.0

        # Generate HTML diff
        diff_html = self._generate_html_diff(lines1, lines2)

        result = TextDiff(
            doc1_id=doc1_id,
            doc2_id=doc2_id,
            additions=additions,
            deletions=deletions,
            modifications=modifications,
            unchanged_ratio=unchanged_ratio,
            diff_html=diff_html
        )

        print(f"✅ Diff complete")
        print(f"   Additions: {len(additions)}")
        print(f"   Deletions: {len(deletions)}")
        print(f"   Modifications: {len(modifications)}")
        print(f"   Unchanged: {unchanged_ratio:.1%}")

        return result

    def _calculate_lexical_similarity(self, text1: str, text2: str) -> float:
        """Calculate lexical similarity using word overlap"""
        # Tokenize
        words1 = set(self._tokenize(text1))
        words2 = set(self._tokenize(text2))

        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        if not self.embedding_manager:
            return 0.0

        # Truncate texts if too long
        max_length = 2000
        text1 = text1[:max_length]
        text2 = text2[:max_length]

        # Get embeddings
        emb1 = self.embedding_manager.embed_text(text1)
        emb2 = self.embedding_manager.embed_text(text2)

        # Compute cosine similarity
        similarity = self.embedding_manager.compute_similarity(emb1, emb2)

        return similarity

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase
        text = text.lower()

        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text)

        # Filter stop words and short words
        words = [
            word for word in words
            if word not in self.stop_words and len(word) >= 3
        ]

        return words

    def _find_common_terms(
        self,
        documents: Dict[str, str],
        top_k: int = 50
    ) -> List[Tuple[str, int]]:
        """Find terms common across documents"""
        # Tokenize all documents
        all_words = []
        for text in documents.values():
            all_words.extend(self._tokenize(text))

        # Count frequency
        counter = Counter(all_words)

        # Filter: only terms appearing in multiple documents
        common_terms = []
        for term, count in counter.most_common():
            # Check if term appears in multiple documents
            doc_count = sum(
                1 for text in documents.values()
                if term in self._tokenize(text)
            )

            if doc_count >= 2:  # At least 2 documents
                common_terms.append((term, count))

            if len(common_terms) >= top_k:
                break

        return common_terms

    def _find_unique_terms(
        self,
        documents: Dict[str, str],
        top_k: int = 20
    ) -> Dict[str, List[str]]:
        """Find unique terms for each document"""
        # Tokenize all documents
        doc_words = {}
        for doc_id, text in documents.items():
            doc_words[doc_id] = set(self._tokenize(text))

        # Find unique terms
        unique_terms = {}

        for doc_id, words in doc_words.items():
            # Get all other words
            other_words = set()
            for other_id, other_set in doc_words.items():
                if other_id != doc_id:
                    other_words.update(other_set)

            # Find unique
            unique = words - other_words

            # Get top by frequency
            text = documents[doc_id]
            word_freq = Counter(self._tokenize(text))
            unique_sorted = sorted(
                unique,
                key=lambda w: word_freq[w],
                reverse=True
            )

            unique_terms[doc_id] = unique_sorted[:top_k]

        return unique_terms

    def _generate_comparison_summary(
        self,
        documents: Dict[str, str],
        similarity_scores: Dict[Tuple[str, str], float],
        common_terms: List[Tuple[str, int]]
    ) -> str:
        """Generate LLM-based comparison summary"""
        if not self.llm_manager:
            return "LLM summary not available"

        # Build prompt
        doc_previews = []
        for doc_id, text in documents.items():
            preview = text[:300] + "..." if len(text) > 300 else text
            doc_previews.append(f"Document {doc_id}:\n{preview}")

        similarity_text = "\n".join([
            f"{doc1} vs {doc2}: {score:.1%} similar"
            for (doc1, doc2), score in list(similarity_scores.items())[:5]
        ])

        common_terms_text = ", ".join([term for term, _ in common_terms[:10]])

        prompt = f"""Compare the following documents and provide a brief summary of their similarities and differences.

Documents:
{chr(10).join(doc_previews)}

Similarity Scores:
{similarity_text}

Common Terms: {common_terms_text}

Please provide a 2-3 sentence comparison summary:"""

        # Generate summary
        response = self.llm_manager.generate(
            query=prompt,
            llm_type=LLMType.OLLAMA,  # Use local for privacy
            temperature=0.5
        )

        return response.get("text", "Could not generate summary")

    def _generate_html_diff(self, lines1: List[str], lines2: List[str]) -> str:
        """Generate HTML diff visualization"""
        differ = difflib.HtmlDiff()
        html = differ.make_table(
            lines1,
            lines2,
            fromdesc="Document 1",
            todesc="Document 2",
            context=True,
            numlines=3
        )
        return html


class SimilarityAnalyzer:
    """
    Analyzes document similarity at different levels
    """

    def __init__(self):
        """Initialize similarity analyzer"""
        try:
            self.embedding_manager = EmbeddingManager()
        except Exception as e:
            print(f"⚠️ Could not initialize embeddings: {e}")
            self.embedding_manager = None

        print(f"✅ Similarity Analyzer initialized")

    def find_similar_documents(
        self,
        query_text: str,
        document_texts: Dict[str, str],
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to query

        Args:
            query_text: Query text
            document_texts: Dictionary of doc_id to text
            top_k: Number of results
            threshold: Minimum similarity threshold

        Returns:
            List of similar documents with scores
        """
        if not self.embedding_manager:
            print("⚠️ Embeddings not available")
            return []

        print(f"\n🔍 Finding similar documents...")

        # Get query embedding
        query_emb = self.embedding_manager.embed_text(query_text[:2000])

        # Calculate similarities
        similarities = []

        for doc_id, text in document_texts.items():
            doc_emb = self.embedding_manager.embed_text(text[:2000])
            similarity = self.embedding_manager.compute_similarity(query_emb, doc_emb)

            if similarity >= threshold:
                similarities.append({
                    "doc_id": doc_id,
                    "similarity": similarity,
                    "preview": text[:200] + "..." if len(text) > 200 else text
                })

        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        results = similarities[:top_k]

        print(f"✅ Found {len(results)} similar documents")

        return results

    def cluster_documents(
        self,
        documents: Dict[str, str],
        num_clusters: int = 3
    ) -> Dict[int, List[str]]:
        """
        Cluster documents by similarity

        Args:
            documents: Dictionary of doc_id to text
            num_clusters: Number of clusters

        Returns:
            Dictionary mapping cluster_id to list of doc_ids
        """
        if not self.embedding_manager:
            print("⚠️ Embeddings not available")
            return {}

        print(f"\n📦 Clustering {len(documents)} documents into {num_clusters} clusters...")

        # Get embeddings for all documents
        embeddings = []
        doc_ids = []

        for doc_id, text in documents.items():
            emb = self.embedding_manager.embed_text(text[:2000])
            embeddings.append(emb)
            doc_ids.append(doc_id)

        # Simple k-means clustering (using numpy)
        import numpy as np

        embeddings_array = np.array(embeddings)

        # Initialize centroids randomly
        indices = np.random.choice(len(embeddings), num_clusters, replace=False)
        centroids = embeddings_array[indices]

        # Iterate
        max_iters = 10
        for _ in range(max_iters):
            # Assign to nearest centroid
            distances = np.zeros((len(embeddings), num_clusters))

            for i, emb in enumerate(embeddings_array):
                for j, centroid in enumerate(centroids):
                    distances[i, j] = np.linalg.norm(emb - centroid)

            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = []
            for j in range(num_clusters):
                cluster_points = embeddings_array[assignments == j]
                if len(cluster_points) > 0:
                    new_centroids.append(np.mean(cluster_points, axis=0))
                else:
                    new_centroids.append(centroids[j])

            centroids = np.array(new_centroids)

        # Build result
        clusters = {}
        for cluster_id in range(num_clusters):
            cluster_docs = [
                doc_ids[i] for i, assignment in enumerate(assignments)
                if assignment == cluster_id
            ]
            clusters[cluster_id] = cluster_docs

        print(f"✅ Clustering complete")
        for cluster_id, docs in clusters.items():
            print(f"   Cluster {cluster_id}: {len(docs)} documents")

        return clusters


# Convenience functions
def compare_two_documents(
    doc1_text: str,
    doc2_text: str,
    doc1_id: str = "doc1",
    doc2_id: str = "doc2"
) -> ComparisonResult:
    """Quick comparison of two documents"""
    comparator = DocumentComparator()
    return comparator.compare_documents({
        doc1_id: doc1_text,
        doc2_id: doc2_text
    })


def generate_diff(
    doc1_text: str,
    doc2_text: str,
    doc1_id: str = "doc1",
    doc2_id: str = "doc2"
) -> TextDiff:
    """Quick diff generation"""
    comparator = DocumentComparator()
    return comparator.diff_documents(doc1_id, doc1_text, doc2_id, doc2_text)


if __name__ == "__main__":
    # Test document comparison
    print("Document Comparison Test")
    print("=" * 50)

    test_docs = {
        "doc1": """
        Machine learning is a subset of artificial intelligence that focuses on
        learning from data. It uses algorithms to identify patterns and make predictions.
        Common applications include image recognition, natural language processing, and
        recommendation systems.
        """,
        "doc2": """
        Artificial intelligence encompasses machine learning and deep learning.
        Machine learning algorithms learn from data to make predictions and decisions.
        Popular use cases include computer vision, speech recognition, and
        personalized recommendations.
        """,
        "doc3": """
        Database systems store and manage structured data efficiently.
        They use SQL queries for data retrieval and manipulation.
        Common database types include relational, NoSQL, and graph databases.
        """
    }

    comparator = DocumentComparator(use_embeddings=False, use_llm=False)

    # Compare documents
    result = comparator.compare_documents(test_docs, include_llm_summary=False)

    print(f"\n📊 Comparison Results:")
    print(f"   Documents: {result.doc_ids}")
    print(f"   Average similarity: {result.metadata['avg_similarity']:.2%}")

    print(f"\n🔤 Common terms:")
    for term, count in result.common_terms[:10]:
        print(f"   - {term}: {count}")

    print(f"\n🔍 Unique terms:")
    for doc_id, terms in result.unique_terms.items():
        print(f"   {doc_id}: {', '.join(terms[:5])}")

    # Test diff
    print(f"\n\n{'='*50}")
    print("Diff Test")
    print("=" * 50)

    diff = comparator.diff_documents("doc1", test_docs["doc1"], "doc2", test_docs["doc2"])

    print(f"\n📝 Diff Results:")
    print(f"   Additions: {len(diff.additions)}")
    print(f"   Deletions: {len(diff.deletions)}")
    print(f"   Modifications: {len(diff.modifications)}")
    print(f"   Unchanged: {diff.unchanged_ratio:.1%}")
