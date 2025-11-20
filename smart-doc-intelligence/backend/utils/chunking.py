"""
Document Chunking Module
Implements various text chunking strategies for RAG
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


class DocumentChunker:
    """
    Handles text chunking with various strategies
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
    ):
        """
        Initialize document chunker

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_by_fixed_size(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split text into fixed-size chunks with overlap

        Args:
            text: Text to chunk
            metadata: Additional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        chunks = []
        chunk_id = 0
        start = 0
        text_length = len(text)
        metadata = metadata or {}

        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)

            # Extract chunk
            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_char=start,
                    end_char=end,
                    metadata={
                        **metadata,
                        "chunking_strategy": "fixed_size",
                        "chunk_size": len(chunk_text),
                    }
                ))
                chunk_id += 1

            # Move start position with overlap
            start = end - self.chunk_overlap

            # Avoid infinite loop
            if start <= 0 or self.chunk_overlap >= self.chunk_size:
                break

        print(f"✅ Created {len(chunks)} fixed-size chunks")
        return chunks

    def chunk_by_paragraph(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split text by paragraphs, combining small paragraphs

        Args:
            text: Text to chunk
            metadata: Additional metadata

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        chunk_id = 0
        current_chunk = []
        current_size = 0
        current_start = 0
        metadata = metadata or {}

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            # If adding this paragraph exceeds chunk size, save current chunk
            if current_size + para_size > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)

                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        start_char=current_start,
                        end_char=current_start + len(chunk_text),
                        metadata={
                            **metadata,
                            "chunking_strategy": "paragraph",
                            "paragraph_count": len(current_chunk),
                        }
                    ))
                    chunk_id += 1

                # Start new chunk with overlap (include last paragraph)
                if self.chunk_overlap > 0 and current_chunk:
                    current_chunk = [current_chunk[-1], para]
                    current_size = len(current_chunk[-2]) + para_size
                else:
                    current_chunk = [para]
                    current_size = para_size

                current_start = current_start + len(chunk_text) - len(current_chunk[-2]) if len(current_chunk) > 1 else current_start + len(chunk_text)

            else:
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_size += para_size

        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    metadata={
                        **metadata,
                        "chunking_strategy": "paragraph",
                        "paragraph_count": len(current_chunk),
                    }
                ))

        print(f"✅ Created {len(chunks)} paragraph-based chunks")
        return chunks

    def chunk_by_sentence(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split text by sentences, combining to reach target size

        Args:
            text: Text to chunk
            metadata: Additional metadata

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        chunk_id = 0
        current_chunk = []
        current_size = 0
        current_start = 0
        metadata = metadata or {}

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            # Check if adding this sentence exceeds chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)

                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        start_char=current_start,
                        end_char=current_start + len(chunk_text),
                        metadata={
                            **metadata,
                            "chunking_strategy": "sentence",
                            "sentence_count": len(current_chunk),
                        }
                    ))
                    chunk_id += 1

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = ' '.join(current_chunk[-2:])
                    current_chunk = current_chunk[-2:] + [sentence]
                    current_size = len(overlap_text) + sentence_size
                else:
                    current_chunk = [sentence]
                    current_size = sentence_size

                current_start = current_start + len(chunk_text) - len(overlap_text) if self.chunk_overlap > 0 else current_start + len(chunk_text)

            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    metadata={
                        **metadata,
                        "chunking_strategy": "sentence",
                        "sentence_count": len(current_chunk),
                    }
                ))

        print(f"✅ Created {len(chunks)} sentence-based chunks")
        return chunks

    def chunk_by_markdown_section(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split markdown text by sections (headers)

        Args:
            text: Markdown text to chunk
            metadata: Additional metadata

        Returns:
            List of Chunk objects
        """
        if not text:
            return []

        # Split by markdown headers
        sections = re.split(r'\n(#{1,6}\s+.*)\n', text)

        chunks = []
        chunk_id = 0
        current_section = []
        current_size = 0
        current_start = 0
        current_header = None
        metadata = metadata or {}

        for i, section in enumerate(sections):
            # Check if this is a header
            if re.match(r'^#{1,6}\s+', section):
                # Save previous section if exists
                if current_section:
                    chunk_text = '\n'.join(current_section)

                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(Chunk(
                            text=chunk_text,
                            chunk_id=chunk_id,
                            start_char=current_start,
                            end_char=current_start + len(chunk_text),
                            metadata={
                                **metadata,
                                "chunking_strategy": "markdown_section",
                                "section_header": current_header,
                            }
                        ))
                        chunk_id += 1

                # Start new section
                current_header = section.strip()
                current_section = [section]
                current_size = len(section)
                current_start = sum(len(s) for s in sections[:i])

            else:
                # Add content to current section
                section = section.strip()
                if section:
                    current_section.append(section)
                    current_size += len(section)

        # Add final section
        if current_section:
            chunk_text = '\n'.join(current_section)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    metadata={
                        **metadata,
                        "chunking_strategy": "markdown_section",
                        "section_header": current_header,
                    }
                ))

        print(f"✅ Created {len(chunks)} markdown section chunks")
        return chunks

    def chunk_document(
        self,
        text: str,
        strategy: str = "paragraph",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk document using specified strategy

        Args:
            text: Text to chunk
            strategy: Chunking strategy ('fixed', 'paragraph', 'sentence', 'markdown')
            metadata: Additional metadata

        Returns:
            List of Chunk objects
        """
        strategies = {
            "fixed": self.chunk_by_fixed_size,
            "paragraph": self.chunk_by_paragraph,
            "sentence": self.chunk_by_sentence,
            "markdown": self.chunk_by_markdown_section,
        }

        chunker = strategies.get(strategy, self.chunk_by_paragraph)
        return chunker(text, metadata)

    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Get statistics about chunks

        Args:
            chunks: List of Chunk objects

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {"chunk_count": 0}

        sizes = [len(chunk.text) for chunk in chunks]

        return {
            "chunk_count": len(chunks),
            "total_characters": sum(sizes),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "strategies_used": list(set(chunk.metadata.get("chunking_strategy", "unknown") for chunk in chunks)),
        }


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    strategy: str = "paragraph"
) -> List[str]:
    """
    Quick function to chunk text and return list of strings

    Args:
        text: Text to chunk
        chunk_size: Target chunk size
        overlap: Chunk overlap
        strategy: Chunking strategy

    Returns:
        List of chunk strings
    """
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = chunker.chunk_document(text, strategy=strategy)
    return [chunk.text for chunk in chunks]


if __name__ == "__main__":
    # Test chunking
    print("Document Chunker Test")
    print("=" * 50)

    sample_text = """
# Introduction

This is a sample document with multiple paragraphs. It demonstrates the chunking functionality.

## Section 1

First paragraph in section 1. This paragraph contains multiple sentences. Each sentence adds to the content.

Second paragraph in section 1. More content here.

## Section 2

This is section 2 content. It has different information.

Final paragraph with concluding remarks.
    """

    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)

    print("\n1. Paragraph chunking:")
    chunks = chunker.chunk_by_paragraph(sample_text)
    for chunk in chunks:
        print(f"  Chunk {chunk.chunk_id}: {len(chunk.text)} chars")

    print(f"\n2. Markdown section chunking:")
    chunks = chunker.chunk_by_markdown_section(sample_text)
    for chunk in chunks:
        print(f"  Chunk {chunk.chunk_id}: {chunk.metadata.get('section_header', 'No header')}")

    print(f"\nStatistics:")
    stats = chunker.get_chunk_statistics(chunks)
    for key, value in stats.items():
        print(f"  {key}: {value}")
