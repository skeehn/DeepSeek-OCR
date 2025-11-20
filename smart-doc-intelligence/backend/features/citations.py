"""
Citation Generator
Generates citations in various academic styles (APA, MLA, Chicago, IEEE)
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class CitationStyle(Enum):
    """Citation style options"""
    APA = "apa"  # American Psychological Association
    MLA = "mla"  # Modern Language Association
    CHICAGO = "chicago"  # Chicago Manual of Style
    IEEE = "ieee"  # Institute of Electrical and Electronics Engineers
    HARVARD = "harvard"  # Harvard referencing


class DocumentType(Enum):
    """Document type options"""
    BOOK = "book"
    ARTICLE = "article"
    WEBSITE = "website"
    JOURNAL = "journal"
    CONFERENCE = "conference"
    THESIS = "thesis"
    REPORT = "report"
    DATASET = "dataset"


@dataclass
class CitationMetadata:
    """Metadata for generating citations"""
    doc_type: DocumentType
    title: str
    authors: List[str]
    year: Optional[int] = None
    publisher: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[int] = None
    issue: Optional[int] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    access_date: Optional[str] = None
    edition: Optional[str] = None
    city: Optional[str] = None
    institution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "doc_type": self.doc_type.value,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "publisher": self.publisher,
            "journal": self.journal,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "doi": self.doi,
            "url": self.url,
            "access_date": self.access_date,
            "edition": self.edition,
            "city": self.city,
            "institution": self.institution,
        }


class APACitationGenerator:
    """
    Generates citations in APA style (7th edition)
    """

    def __init__(self):
        """Initialize APA citation generator"""
        print(f"✅ APA Citation Generator initialized")

    def generate(self, metadata: CitationMetadata) -> str:
        """
        Generate APA citation

        Args:
            metadata: Citation metadata

        Returns:
            Formatted citation string
        """
        doc_type = metadata.doc_type

        if doc_type == DocumentType.BOOK:
            return self._cite_book(metadata)
        elif doc_type == DocumentType.ARTICLE:
            return self._cite_article(metadata)
        elif doc_type == DocumentType.JOURNAL:
            return self._cite_journal(metadata)
        elif doc_type == DocumentType.WEBSITE:
            return self._cite_website(metadata)
        elif doc_type == DocumentType.CONFERENCE:
            return self._cite_conference(metadata)
        else:
            return self._cite_generic(metadata)

    def _format_authors(self, authors: List[str]) -> str:
        """Format author names in APA style"""
        if not authors:
            return ""

        if len(authors) == 1:
            return self._format_author_name(authors[0])
        elif len(authors) == 2:
            return f"{self._format_author_name(authors[0])}, & {self._format_author_name(authors[1])}"
        else:
            formatted = [self._format_author_name(a) for a in authors[:20]]
            if len(authors) > 20:
                formatted.append("...")
            formatted[-1] = f"& {formatted[-1]}"
            return ", ".join(formatted)

    def _format_author_name(self, name: str) -> str:
        """Format single author name (Last, F. M.)"""
        parts = name.split()
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[-1]}, {parts[0][0]}."
        else:
            # Assume: First Middle Last
            initials = " ".join([p[0] + "." for p in parts[:-1]])
            return f"{parts[-1]}, {initials}"

    def _cite_book(self, metadata: CitationMetadata) -> str:
        """Generate book citation"""
        authors = self._format_authors(metadata.authors)
        year = f"({metadata.year})" if metadata.year else "(n.d.)"
        title = f"*{metadata.title}*"

        citation = f"{authors} {year}. {title}."

        if metadata.edition:
            citation += f" ({metadata.edition} ed.)."

        if metadata.publisher:
            citation += f" {metadata.publisher}."

        if metadata.doi:
            citation += f" https://doi.org/{metadata.doi}"

        return citation

    def _cite_article(self, metadata: CitationMetadata) -> str:
        """Generate article citation"""
        authors = self._format_authors(metadata.authors)
        year = f"({metadata.year})" if metadata.year else "(n.d.)"
        title = metadata.title

        citation = f"{authors} {year}. {title}."

        if metadata.url:
            citation += f" Retrieved from {metadata.url}"

        return citation

    def _cite_journal(self, metadata: CitationMetadata) -> str:
        """Generate journal article citation"""
        authors = self._format_authors(metadata.authors)
        year = f"({metadata.year})" if metadata.year else "(n.d.)"
        title = metadata.title
        journal = f"*{metadata.journal}*" if metadata.journal else ""

        citation = f"{authors} {year}. {title}. {journal}"

        if metadata.volume:
            citation += f", *{metadata.volume}*"
            if metadata.issue:
                citation += f"({metadata.issue})"

        if metadata.pages:
            citation += f", {metadata.pages}"

        citation += "."

        if metadata.doi:
            citation += f" https://doi.org/{metadata.doi}"
        elif metadata.url:
            citation += f" {metadata.url}"

        return citation

    def _cite_website(self, metadata: CitationMetadata) -> str:
        """Generate website citation"""
        authors = self._format_authors(metadata.authors) if metadata.authors else metadata.publisher or "Author"
        year = f"({metadata.year})" if metadata.year else "(n.d.)"
        title = f"*{metadata.title}*"

        citation = f"{authors} {year}. {title}."

        if metadata.url:
            citation += f" {metadata.url}"

        if metadata.access_date:
            citation += f" (accessed {metadata.access_date})"

        return citation

    def _cite_conference(self, metadata: CitationMetadata) -> str:
        """Generate conference paper citation"""
        authors = self._format_authors(metadata.authors)
        year = f"({metadata.year})" if metadata.year else "(n.d.)"
        title = metadata.title
        conference = f"*{metadata.publisher}*" if metadata.publisher else ""

        citation = f"{authors} {year}. {title}. In {conference}"

        if metadata.pages:
            citation += f" (pp. {metadata.pages})"

        citation += "."

        return citation

    def _cite_generic(self, metadata: CitationMetadata) -> str:
        """Generate generic citation"""
        authors = self._format_authors(metadata.authors)
        year = f"({metadata.year})" if metadata.year else "(n.d.)"
        title = f"*{metadata.title}*"

        citation = f"{authors} {year}. {title}."

        return citation


class MLACitationGenerator:
    """
    Generates citations in MLA style (9th edition)
    """

    def __init__(self):
        """Initialize MLA citation generator"""
        print(f"✅ MLA Citation Generator initialized")

    def generate(self, metadata: CitationMetadata) -> str:
        """Generate MLA citation"""
        doc_type = metadata.doc_type

        if doc_type == DocumentType.BOOK:
            return self._cite_book(metadata)
        elif doc_type == DocumentType.JOURNAL:
            return self._cite_journal(metadata)
        elif doc_type == DocumentType.WEBSITE:
            return self._cite_website(metadata)
        else:
            return self._cite_generic(metadata)

    def _format_authors(self, authors: List[str]) -> str:
        """Format author names in MLA style"""
        if not authors:
            return ""

        if len(authors) == 1:
            return self._format_author_name(authors[0])
        elif len(authors) == 2:
            return f"{self._format_author_name(authors[0])}, and {self._format_author_name(authors[1], first=False)}"
        else:
            return f"{self._format_author_name(authors[0])}, et al."

    def _format_author_name(self, name: str, first: bool = True) -> str:
        """Format single author name"""
        parts = name.split()
        if first:
            # First author: Last, First
            if len(parts) >= 2:
                return f"{parts[-1]}, {' '.join(parts[:-1])}"
            return name
        else:
            # Subsequent: First Last
            return name

    def _cite_book(self, metadata: CitationMetadata) -> str:
        """Generate book citation"""
        authors = self._format_authors(metadata.authors)
        title = f"*{metadata.title}*"

        citation = f"{authors}. {title}."

        if metadata.publisher:
            citation += f" {metadata.publisher}"

        if metadata.year:
            citation += f", {metadata.year}"

        citation += "."

        return citation

    def _cite_journal(self, metadata: CitationMetadata) -> str:
        """Generate journal article citation"""
        authors = self._format_authors(metadata.authors)
        title = f'"{metadata.title}"'
        journal = f"*{metadata.journal}*" if metadata.journal else ""

        citation = f"{authors}. {title}. {journal}"

        if metadata.volume:
            citation += f", vol. {metadata.volume}"
            if metadata.issue:
                citation += f", no. {metadata.issue}"

        if metadata.year:
            citation += f", {metadata.year}"

        if metadata.pages:
            citation += f", pp. {metadata.pages}"

        citation += "."

        if metadata.doi:
            citation += f" DOI: {metadata.doi}."

        return citation

    def _cite_website(self, metadata: CitationMetadata) -> str:
        """Generate website citation"""
        authors = self._format_authors(metadata.authors) if metadata.authors else ""
        title = f'"{metadata.title}"'
        website = f"*{metadata.publisher}*" if metadata.publisher else ""

        citation = f"{authors}. {title}. {website}"

        if metadata.year:
            citation += f", {metadata.year}"

        citation += "."

        if metadata.url:
            citation += f" {metadata.url}."

        if metadata.access_date:
            citation += f" Accessed {metadata.access_date}."

        return citation

    def _cite_generic(self, metadata: CitationMetadata) -> str:
        """Generate generic citation"""
        authors = self._format_authors(metadata.authors)
        title = f"*{metadata.title}*"

        citation = f"{authors}. {title}"

        if metadata.year:
            citation += f". {metadata.year}"

        citation += "."

        return citation


class ChicagoCitationGenerator:
    """
    Generates citations in Chicago style (17th edition)
    """

    def __init__(self):
        """Initialize Chicago citation generator"""
        print(f"✅ Chicago Citation Generator initialized")

    def generate(self, metadata: CitationMetadata) -> str:
        """Generate Chicago citation"""
        doc_type = metadata.doc_type

        if doc_type == DocumentType.BOOK:
            return self._cite_book(metadata)
        elif doc_type == DocumentType.JOURNAL:
            return self._cite_journal(metadata)
        elif doc_type == DocumentType.WEBSITE:
            return self._cite_website(metadata)
        else:
            return self._cite_generic(metadata)

    def _format_authors(self, authors: List[str]) -> str:
        """Format author names in Chicago style"""
        if not authors:
            return ""

        if len(authors) == 1:
            return self._format_author_name(authors[0])
        elif len(authors) == 2:
            return f"{self._format_author_name(authors[0])} and {self._format_author_name(authors[1], first=False)}"
        elif len(authors) == 3:
            return f"{self._format_author_name(authors[0])}, {self._format_author_name(authors[1], first=False)}, and {self._format_author_name(authors[2], first=False)}"
        else:
            return f"{self._format_author_name(authors[0])} et al."

    def _format_author_name(self, name: str, first: bool = True) -> str:
        """Format single author name"""
        parts = name.split()
        if first:
            # First author: Last, First
            if len(parts) >= 2:
                return f"{parts[-1]}, {' '.join(parts[:-1])}"
            return name
        else:
            # Subsequent: First Last
            return name

    def _cite_book(self, metadata: CitationMetadata) -> str:
        """Generate book citation"""
        authors = self._format_authors(metadata.authors)
        title = f"*{metadata.title}*"

        citation = f"{authors}. {title}."

        if metadata.city and metadata.publisher:
            citation += f" {metadata.city}: {metadata.publisher}"
        elif metadata.publisher:
            citation += f" {metadata.publisher}"

        if metadata.year:
            citation += f", {metadata.year}"

        citation += "."

        return citation

    def _cite_journal(self, metadata: CitationMetadata) -> str:
        """Generate journal article citation"""
        authors = self._format_authors(metadata.authors)
        title = f'"{metadata.title}"'
        journal = f"*{metadata.journal}*" if metadata.journal else ""

        citation = f"{authors}. {title}. {journal}"

        if metadata.volume:
            citation += f" {metadata.volume}"
            if metadata.issue:
                citation += f", no. {metadata.issue}"

        if metadata.year:
            citation += f" ({metadata.year})"

        if metadata.pages:
            citation += f": {metadata.pages}"

        citation += "."

        return citation

    def _cite_website(self, metadata: CitationMetadata) -> str:
        """Generate website citation"""
        authors = self._format_authors(metadata.authors) if metadata.authors else metadata.publisher or ""
        title = f'"{metadata.title}"'

        citation = f"{authors}. {title}."

        if metadata.publisher and not metadata.authors:
            citation = f"{title}. {metadata.publisher}."

        if metadata.year:
            citation += f" {metadata.year}."

        if metadata.url:
            citation += f" {metadata.url}."

        return citation

    def _cite_generic(self, metadata: CitationMetadata) -> str:
        """Generate generic citation"""
        authors = self._format_authors(metadata.authors)
        title = f"*{metadata.title}*"

        citation = f"{authors}. {title}"

        if metadata.year:
            citation += f". {metadata.year}"

        citation += "."

        return citation


class CitationGenerator:
    """
    High-level citation generator
    Supports multiple citation styles
    """

    def __init__(self):
        """Initialize citation generator"""
        self.apa = APACitationGenerator()
        self.mla = MLACitationGenerator()
        self.chicago = ChicagoCitationGenerator()

        print(f"✅ Citation Generator initialized")
        print(f"   Styles: APA, MLA, Chicago")

    def generate(
        self,
        metadata: CitationMetadata,
        style: CitationStyle = CitationStyle.APA
    ) -> str:
        """
        Generate citation in specified style

        Args:
            metadata: Citation metadata
            style: Citation style

        Returns:
            Formatted citation string
        """
        if style == CitationStyle.APA:
            return self.apa.generate(metadata)
        elif style == CitationStyle.MLA:
            return self.mla.generate(metadata)
        elif style == CitationStyle.CHICAGO:
            return self.chicago.generate(metadata)
        else:
            # Default to APA
            return self.apa.generate(metadata)

    def generate_multiple_styles(
        self,
        metadata: CitationMetadata,
        styles: Optional[List[CitationStyle]] = None
    ) -> Dict[str, str]:
        """
        Generate citations in multiple styles

        Args:
            metadata: Citation metadata
            styles: List of styles (None = all)

        Returns:
            Dictionary mapping style to citation
        """
        if styles is None:
            styles = [CitationStyle.APA, CitationStyle.MLA, CitationStyle.CHICAGO]

        citations = {}

        for style in styles:
            citations[style.value] = self.generate(metadata, style)

        return citations

    def generate_bibliography(
        self,
        citations: List[CitationMetadata],
        style: CitationStyle = CitationStyle.APA,
        sort_alphabetically: bool = True
    ) -> str:
        """
        Generate bibliography from multiple citations

        Args:
            citations: List of citation metadata
            style: Citation style
            sort_alphabetically: Sort alphabetically by author

        Returns:
            Formatted bibliography
        """
        # Generate individual citations
        citation_strings = [
            self.generate(metadata, style)
            for metadata in citations
        ]

        # Sort if requested
        if sort_alphabetically:
            citation_strings.sort()

        # Build bibliography
        bibliography = "References\n" + "=" * 50 + "\n\n"
        bibliography += "\n\n".join(citation_strings)

        return bibliography


# Convenience functions
def quick_cite(
    title: str,
    authors: List[str],
    year: int,
    doc_type: DocumentType = DocumentType.ARTICLE,
    style: CitationStyle = CitationStyle.APA,
    **kwargs
) -> str:
    """Quick citation generation"""
    metadata = CitationMetadata(
        doc_type=doc_type,
        title=title,
        authors=authors,
        year=year,
        **kwargs
    )

    generator = CitationGenerator()
    return generator.generate(metadata, style)


if __name__ == "__main__":
    # Test citation generator
    print("Citation Generator Test")
    print("=" * 50)

    # Test data
    metadata = CitationMetadata(
        doc_type=DocumentType.JOURNAL,
        title="Deep Learning for Natural Language Processing",
        authors=["John Smith", "Jane Doe", "Bob Johnson"],
        year=2023,
        journal="Journal of Artificial Intelligence",
        volume=45,
        issue=3,
        pages="123-145",
        doi="10.1234/jai.2023.001"
    )

    generator = CitationGenerator()

    # Generate in multiple styles
    print("\n📚 Citation Examples:\n")

    print("APA Style:")
    print(generator.generate(metadata, CitationStyle.APA))

    print("\n\nMLA Style:")
    print(generator.generate(metadata, CitationStyle.MLA))

    print("\n\nChicago Style:")
    print(generator.generate(metadata, CitationStyle.CHICAGO))

    # Test book citation
    print("\n\n" + "=" * 50)
    book_metadata = CitationMetadata(
        doc_type=DocumentType.BOOK,
        title="Introduction to Machine Learning",
        authors=["Alice Brown"],
        year=2022,
        publisher="Tech Press",
        city="Cambridge",
        edition="3rd"
    )

    print("\nBook Citation (APA):")
    print(generator.generate(book_metadata, CitationStyle.APA))

    # Test website citation
    print("\n\n" + "=" * 50)
    website_metadata = CitationMetadata(
        doc_type=DocumentType.WEBSITE,
        title="Understanding Neural Networks",
        authors=["Charlie Davis"],
        year=2024,
        publisher="AI Education",
        url="https://example.com/neural-networks",
        access_date="January 15, 2024"
    )

    print("\nWebsite Citation (MLA):")
    print(generator.generate(website_metadata, CitationStyle.MLA))
