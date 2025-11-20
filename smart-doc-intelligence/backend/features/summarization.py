"""
Document Summarization Module
Provides multiple summarization strategies: extractive, abstractive, hierarchical
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import math

from backend.llm.query_router import DualLLMManager, LLMType


class SummaryStyle(Enum):
    """Summarization style options"""
    PARAGRAPH = "paragraph"
    BULLET_POINTS = "bullet_points"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    SIMPLE = "simple"


class SummaryLength(Enum):
    """Summary length options"""
    VERY_SHORT = "very_short"  # 1-2 sentences
    SHORT = "short"  # 3-5 sentences
    MEDIUM = "medium"  # 1-2 paragraphs
    LONG = "long"  # Multiple paragraphs
    DETAILED = "detailed"  # Comprehensive summary


@dataclass
class Summary:
    """Represents a document summary"""
    text: str
    style: SummaryStyle
    length: SummaryLength
    method: str  # extractive, abstractive, hierarchical
    key_points: List[str]
    word_count: int
    compression_ratio: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "style": self.style.value,
            "length": self.length.value,
            "method": self.method,
            "key_points": self.key_points,
            "word_count": self.word_count,
            "compression_ratio": self.compression_ratio,
            "metadata": self.metadata,
        }


class ExtractiveSummarizer:
    """
    Extractive summarization: selects key sentences from document
    Uses sentence scoring based on term frequency and position
    """

    def __init__(self):
        """Initialize extractive summarizer"""
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those'
        }

        print(f"✅ Extractive Summarizer initialized")

    def summarize(
        self,
        text: str,
        num_sentences: int = 5,
        prioritize_beginning: bool = True
    ) -> List[str]:
        """
        Extract key sentences from text

        Args:
            text: Input text
            num_sentences: Number of sentences to extract
            prioritize_beginning: Give higher scores to early sentences

        Returns:
            List of key sentences
        """
        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= num_sentences:
            return sentences

        # Calculate term frequencies
        word_freq = self._calculate_word_frequencies(text)

        # Score sentences
        sentence_scores = []

        for idx, sentence in enumerate(sentences):
            # Base score from word frequencies
            score = self._score_sentence(sentence, word_freq)

            # Position bonus (early sentences often more important)
            if prioritize_beginning:
                position_bonus = 1.0 - (idx / len(sentences)) * 0.3
                score *= position_bonus

            # Length penalty (very short sentences less informative)
            word_count = len(sentence.split())
            if word_count < 5:
                score *= 0.5

            sentence_scores.append((idx, sentence, score))

        # Sort by score
        sentence_scores.sort(key=lambda x: x[2], reverse=True)

        # Select top sentences
        top_sentences = sentence_scores[:num_sentences]

        # Re-sort by original position
        top_sentences.sort(key=lambda x: x[0])

        return [sent for _, sent, _ in top_sentences]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        text = text.replace('\n', ' ')
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _calculate_word_frequencies(self, text: str) -> Dict[str, float]:
        """Calculate normalized word frequencies"""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        words = [w for w in words if w not in self.stop_words and len(w) >= 3]

        # Count frequencies
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1

        # Normalize
        max_freq = max(word_count.values()) if word_count else 1

        word_freq = {
            word: count / max_freq
            for word, count in word_count.items()
        }

        return word_freq

    def _score_sentence(self, sentence: str, word_freq: Dict[str, float]) -> float:
        """Score sentence based on word frequencies"""
        words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        words = [w for w in words if w not in self.stop_words and len(w) >= 3]

        if not words:
            return 0.0

        # Average word frequency
        score = sum(word_freq.get(word, 0) for word in words) / len(words)

        return score


class AbstractiveSummarizer:
    """
    Abstractive summarization: generates new text using LLM
    Provides more natural, coherent summaries
    """

    def __init__(self, use_cloud: bool = False):
        """
        Initialize abstractive summarizer

        Args:
            use_cloud: Use cloud LLM (Gemini) for better quality
        """
        self.use_cloud = use_cloud

        try:
            self.llm_manager = DualLLMManager(prefer_local=not use_cloud)
        except Exception as e:
            print(f"⚠️ Could not initialize LLM: {e}")
            self.llm_manager = None

        print(f"✅ Abstractive Summarizer initialized")
        print(f"   Using: {'Cloud LLM' if use_cloud else 'Local LLM'}")

    def summarize(
        self,
        text: str,
        style: SummaryStyle = SummaryStyle.PARAGRAPH,
        length: SummaryLength = SummaryLength.MEDIUM,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Generate abstractive summary

        Args:
            text: Input text
            style: Summary style
            length: Summary length
            custom_instructions: Custom summarization instructions

        Returns:
            Generated summary
        """
        if not self.llm_manager:
            return "Error: LLM not available for abstractive summarization"

        # Build prompt
        prompt = self._build_prompt(text, style, length, custom_instructions)

        # Generate summary
        llm_type = LLMType.GEMINI if self.use_cloud else LLMType.OLLAMA

        response = self.llm_manager.generate(
            query=prompt,
            llm_type=llm_type,
            temperature=0.5
        )

        summary_text = response.get("text", "Could not generate summary")

        return summary_text

    def _build_prompt(
        self,
        text: str,
        style: SummaryStyle,
        length: SummaryLength,
        custom_instructions: Optional[str]
    ) -> str:
        """Build summarization prompt"""
        # Truncate text if too long
        max_length = 3000
        if len(text) > max_length:
            text = text[:max_length] + "\n\n[Document truncated...]"

        # Style instructions
        style_instructions = {
            SummaryStyle.PARAGRAPH: "Write the summary as a coherent paragraph.",
            SummaryStyle.BULLET_POINTS: "Write the summary as bullet points highlighting key information.",
            SummaryStyle.EXECUTIVE: "Write an executive summary suitable for business decision-makers.",
            SummaryStyle.TECHNICAL: "Write a technical summary preserving important details and terminology.",
            SummaryStyle.SIMPLE: "Write a simple, easy-to-understand summary suitable for general audiences.",
        }

        # Length instructions
        length_instructions = {
            SummaryLength.VERY_SHORT: "Keep it very brief (1-2 sentences).",
            SummaryLength.SHORT: "Keep it short (3-5 sentences).",
            SummaryLength.MEDIUM: "Write a medium-length summary (1-2 paragraphs).",
            SummaryLength.LONG: "Write a comprehensive summary (2-3 paragraphs).",
            SummaryLength.DETAILED: "Write a detailed summary covering all important aspects.",
        }

        prompt = f"""Summarize the following document.

{style_instructions.get(style, '')}
{length_instructions.get(length, '')}
{custom_instructions if custom_instructions else ''}

Document:
{text}

Summary:"""

        return prompt


class HierarchicalSummarizer:
    """
    Hierarchical summarization: breaks long documents into chunks,
    summarizes each chunk, then summarizes the summaries
    """

    def __init__(self, use_cloud: bool = False):
        """
        Initialize hierarchical summarizer

        Args:
            use_cloud: Use cloud LLM
        """
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer(use_cloud=use_cloud)

        print(f"✅ Hierarchical Summarizer initialized")

    def summarize(
        self,
        text: str,
        chunk_size: int = 2000,
        final_style: SummaryStyle = SummaryStyle.PARAGRAPH,
        final_length: SummaryLength = SummaryLength.MEDIUM
    ) -> Dict[str, Any]:
        """
        Generate hierarchical summary

        Args:
            text: Input text (can be very long)
            chunk_size: Size of each chunk
            final_style: Style for final summary
            final_length: Length for final summary

        Returns:
            Dictionary with final summary and intermediate summaries
        """
        print(f"\n📄 Hierarchical summarization...")

        # Split into chunks
        chunks = self._chunk_text(text, chunk_size)
        print(f"   Split into {len(chunks)} chunks")

        # Summarize each chunk
        chunk_summaries = []

        for i, chunk in enumerate(chunks):
            print(f"   Summarizing chunk {i+1}/{len(chunks)}...")

            # Use extractive for intermediate summaries (faster)
            sentences = self.extractive.summarize(chunk, num_sentences=3)
            chunk_summary = " ".join(sentences)
            chunk_summaries.append(chunk_summary)

        # Combine chunk summaries
        combined = "\n\n".join(chunk_summaries)

        # Generate final summary
        print(f"   Generating final summary...")
        final_summary = self.abstractive.summarize(
            combined,
            style=final_style,
            length=final_length
        )

        result = {
            "final_summary": final_summary,
            "chunk_summaries": chunk_summaries,
            "num_chunks": len(chunks),
            "original_length": len(text),
            "compressed_length": len(final_summary),
        }

        print(f"✅ Hierarchical summarization complete")

        return result

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []

        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1

            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        # Add remaining
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class DocumentSummarizer:
    """
    High-level document summarizer
    Automatically selects best strategy based on document characteristics
    """

    def __init__(self, use_cloud: bool = False):
        """
        Initialize document summarizer

        Args:
            use_cloud: Use cloud LLM for abstractive methods
        """
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer(use_cloud=use_cloud)
        self.hierarchical = HierarchicalSummarizer(use_cloud=use_cloud)

        print(f"✅ Document Summarizer initialized")

    def summarize(
        self,
        text: str,
        style: SummaryStyle = SummaryStyle.PARAGRAPH,
        length: SummaryLength = SummaryLength.MEDIUM,
        method: Optional[str] = None
    ) -> Summary:
        """
        Summarize document with automatic method selection

        Args:
            text: Input text
            style: Summary style
            length: Summary length
            method: Force specific method (extractive, abstractive, hierarchical)

        Returns:
            Summary object
        """
        print(f"\n📝 Summarizing document...")
        print(f"   Length: {len(text)} characters")
        print(f"   Style: {style.value}")
        print(f"   Target length: {length.value}")

        original_word_count = len(text.split())

        # Auto-select method if not specified
        if method is None:
            if len(text) > 10000:
                method = "hierarchical"
            elif style == SummaryStyle.BULLET_POINTS:
                method = "extractive"
            else:
                method = "abstractive"

        print(f"   Method: {method}")

        # Generate summary
        if method == "extractive":
            summary_text, key_points = self._extractive_summary(text, length)

        elif method == "abstractive":
            summary_text = self.abstractive.summarize(text, style, length)
            key_points = self._extract_key_points(summary_text)

        elif method == "hierarchical":
            result = self.hierarchical.summarize(text, final_style=style, final_length=length)
            summary_text = result["final_summary"]
            key_points = self._extract_key_points(summary_text)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate metrics
        summary_word_count = len(summary_text.split())
        compression_ratio = summary_word_count / original_word_count if original_word_count > 0 else 0.0

        summary = Summary(
            text=summary_text,
            style=style,
            length=length,
            method=method,
            key_points=key_points,
            word_count=summary_word_count,
            compression_ratio=compression_ratio,
            metadata={
                "original_word_count": original_word_count,
                "original_char_count": len(text),
            }
        )

        print(f"✅ Summarization complete")
        print(f"   Compression: {compression_ratio:.1%}")
        print(f"   Word count: {summary_word_count}")

        return summary

    def _extractive_summary(
        self,
        text: str,
        length: SummaryLength
    ) -> tuple[str, List[str]]:
        """Generate extractive summary"""
        # Map length to number of sentences
        length_mapping = {
            SummaryLength.VERY_SHORT: 2,
            SummaryLength.SHORT: 4,
            SummaryLength.MEDIUM: 6,
            SummaryLength.LONG: 10,
            SummaryLength.DETAILED: 15,
        }

        num_sentences = length_mapping.get(length, 5)

        sentences = self.extractive.summarize(text, num_sentences=num_sentences)
        summary_text = " ".join(sentences)

        # Key points are the sentences themselves
        key_points = sentences[:5]  # Top 5

        return summary_text, key_points

    def _extract_key_points(self, summary_text: str) -> List[str]:
        """Extract key points from summary"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', summary_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Return top sentences as key points
        return sentences[:5]

    def summarize_multiple(
        self,
        documents: Dict[str, str],
        style: SummaryStyle = SummaryStyle.PARAGRAPH,
        length: SummaryLength = SummaryLength.SHORT
    ) -> Dict[str, Summary]:
        """
        Summarize multiple documents

        Args:
            documents: Dictionary of doc_id to text
            style: Summary style
            length: Summary length

        Returns:
            Dictionary of doc_id to Summary
        """
        print(f"\n📚 Summarizing {len(documents)} documents...")

        summaries = {}

        for doc_id, text in documents.items():
            print(f"\n   Document: {doc_id}")
            summary = self.summarize(text, style=style, length=length)
            summaries[doc_id] = summary

        print(f"\n✅ All summaries complete")

        return summaries


# Convenience functions
def quick_summary(
    text: str,
    length: SummaryLength = SummaryLength.SHORT,
    use_cloud: bool = False
) -> str:
    """Quick document summary"""
    summarizer = DocumentSummarizer(use_cloud=use_cloud)
    summary = summarizer.summarize(text, length=length)
    return summary.text


def bullet_point_summary(text: str, num_points: int = 5) -> List[str]:
    """Quick bullet point summary"""
    extractive = ExtractiveSummarizer()
    return extractive.summarize(text, num_sentences=num_points)


if __name__ == "__main__":
    # Test summarization
    print("Summarization Test")
    print("=" * 50)

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
    """

    # Test extractive
    print("\n1. Extractive Summarization:")
    extractive = ExtractiveSummarizer()
    sentences = extractive.summarize(test_text, num_sentences=3)
    print("\n".join(sentences))

    # Test abstractive (without LLM)
    print("\n\n2. Abstractive Summarization:")
    print("(Requires LLM - skipping in test)")

    # Test document summarizer
    print("\n\n3. Document Summarizer (Extractive):")
    summarizer = DocumentSummarizer(use_cloud=False)

    try:
        summary = summarizer.summarize(
            test_text,
            style=SummaryStyle.PARAGRAPH,
            length=SummaryLength.SHORT,
            method="extractive"
        )

        print(f"\nSummary: {summary.text}")
        print(f"Method: {summary.method}")
        print(f"Compression: {summary.compression_ratio:.1%}")
        print(f"Word count: {summary.word_count}")
    except Exception as e:
        print(f"Error: {e}")
