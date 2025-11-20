"""
Entity Extraction Module
Extracts named entities, key terms, and relationships from documents
"""
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import Counter

from backend.llm.query_router import DualLLMManager, LLMType


@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    type: str
    confidence: float
    occurrences: int
    context: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "type": self.type,
            "confidence": self.confidence,
            "occurrences": self.occurrences,
            "context": self.context[:3]  # First 3 contexts
        }


class EntityExtractor:
    """
    Extracts entities from documents using pattern matching and LLM
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize entity extractor

        Args:
            use_llm: Use LLM for entity extraction
        """
        self.use_llm = use_llm
        self.llm_manager = None

        if use_llm:
            try:
                self.llm_manager = DualLLMManager(prefer_local=True)
            except Exception as e:
                print(f"⚠️ Could not initialize LLM: {e}")
                self.use_llm = False

        # Regex patterns for common entities
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "url": r'https?://[^\s]+',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            "money": r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            "percentage": r'\b\d+(?:\.\d+)?%',
        }

        print(f"✅ Entity Extractor initialized")
        print(f"   LLM-based extraction: {self.use_llm}")

    def extract_from_text(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> Dict[str, List[Entity]]:
        """
        Extract entities from text

        Args:
            text: Input text
            entity_types: List of entity types to extract (None = all)

        Returns:
            Dictionary mapping entity types to lists of entities
        """
        entities = {}

        # Pattern-based extraction
        pattern_entities = self._extract_with_patterns(text)
        entities.update(pattern_entities)

        # LLM-based extraction
        if self.use_llm and self.llm_manager:
            llm_entities = self._extract_with_llm(text, entity_types)
            # Merge LLM entities
            for ent_type, ent_list in llm_entities.items():
                if ent_type in entities:
                    entities[ent_type].extend(ent_list)
                else:
                    entities[ent_type] = ent_list

        # Filter by requested types
        if entity_types:
            entities = {k: v for k, v in entities.items() if k in entity_types}

        return entities

    def _extract_with_patterns(self, text: str) -> Dict[str, List[Entity]]:
        """Extract entities using regex patterns"""
        entities = {}

        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)

            if matches:
                # Count occurrences
                counter = Counter(matches)

                entity_list = []
                for match, count in counter.most_common():
                    # Find contexts
                    contexts = self._find_contexts(text, match)

                    entity = Entity(
                        text=match,
                        type=entity_type,
                        confidence=1.0,  # Pattern matches are 100% confident
                        occurrences=count,
                        context=contexts
                    )
                    entity_list.append(entity)

                entities[entity_type] = entity_list

        return entities

    def _extract_with_llm(
        self,
        text: str,
        entity_types: Optional[List[str]] = None
    ) -> Dict[str, List[Entity]]:
        """Extract entities using LLM"""
        if not self.llm_manager:
            return {}

        # Build prompt
        types_str = ", ".join(entity_types) if entity_types else "people, organizations, locations, dates, key terms"

        prompt = f"""Extract the following types of entities from the text: {types_str}

Text:
{text[:2000]}

Please list the entities in this exact format:
TYPE: entity1, entity2, entity3

For example:
PERSON: John Smith, Jane Doe
ORGANIZATION: Microsoft, Google
LOCATION: New York, California

Entities:"""

        # Generate with local LLM
        response = self.llm_manager.generate(
            query=prompt,
            llm_type=LLMType.OLLAMA,  # Use local for privacy
            temperature=0.3
        )

        # Parse response
        entities = self._parse_llm_response(response.get("text", ""), text)

        return entities

    def _parse_llm_response(self, response: str, original_text: str) -> Dict[str, List[Entity]]:
        """Parse LLM response to extract entities"""
        entities = {}

        lines = response.strip().split('\n')

        for line in lines:
            if ':' not in line:
                continue

            parts = line.split(':', 1)
            if len(parts) != 2:
                continue

            entity_type = parts[0].strip().lower()
            entity_texts = [e.strip() for e in parts[1].split(',')]

            entity_list = []
            for entity_text in entity_texts:
                if not entity_text or len(entity_text) < 2:
                    continue

                # Count occurrences in original text
                count = original_text.lower().count(entity_text.lower())

                if count > 0:
                    contexts = self._find_contexts(original_text, entity_text)

                    entity = Entity(
                        text=entity_text,
                        type=entity_type,
                        confidence=0.8,  # LLM extractions have lower confidence
                        occurrences=count,
                        context=contexts
                    )
                    entity_list.append(entity)

            if entity_list:
                entities[entity_type] = entity_list

        return entities

    def _find_contexts(self, text: str, entity: str, window: int = 50) -> List[str]:
        """Find contexts where entity appears"""
        contexts = []
        text_lower = text.lower()
        entity_lower = entity.lower()

        start = 0
        while True:
            pos = text_lower.find(entity_lower, start)
            if pos == -1:
                break

            # Extract context window
            context_start = max(0, pos - window)
            context_end = min(len(text), pos + len(entity) + window)
            context = text[context_start:context_end]

            contexts.append(context.strip())

            start = pos + 1

            if len(contexts) >= 5:  # Max 5 contexts
                break

        return contexts

    def extract_key_terms(
        self,
        text: str,
        top_k: int = 20,
        min_length: int = 3
    ) -> List[Tuple[str, int]]:
        """
        Extract key terms (most frequent meaningful words)

        Args:
            text: Input text
            top_k: Number of top terms
            min_length: Minimum term length

        Returns:
            List of (term, frequency) tuples
        """
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        # Filter and count
        filtered_words = [
            word for word in words
            if len(word) >= min_length and word not in stop_words
        ]

        counter = Counter(filtered_words)

        return counter.most_common(top_k)

    def summarize_entities(
        self,
        entities: Dict[str, List[Entity]]
    ) -> Dict[str, Any]:
        """
        Create summary statistics of entities

        Args:
            entities: Dictionary of entities

        Returns:
            Summary statistics
        """
        total_entities = sum(len(ent_list) for ent_list in entities.values())
        total_occurrences = sum(
            sum(ent.occurrences for ent in ent_list)
            for ent_list in entities.values()
        )

        entity_types = list(entities.keys())

        # Most frequent entities
        all_entities = []
        for ent_list in entities.values():
            all_entities.extend(ent_list)

        all_entities.sort(key=lambda x: x.occurrences, reverse=True)

        return {
            "total_entities": total_entities,
            "total_occurrences": total_occurrences,
            "entity_types": entity_types,
            "top_entities": [
                {
                    "text": ent.text,
                    "type": ent.type,
                    "occurrences": ent.occurrences
                }
                for ent in all_entities[:10]
            ]
        }


class DocumentAnalyzer:
    """
    Analyzes documents to extract insights, entities, and structure
    """

    def __init__(self):
        """Initialize document analyzer"""
        self.entity_extractor = EntityExtractor(use_llm=True)

        print(f"✅ Document Analyzer initialized")

    def analyze_document(
        self,
        text: str,
        extract_entities: bool = True,
        extract_key_terms: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive document analysis

        Args:
            text: Document text
            extract_entities: Extract named entities
            extract_key_terms: Extract key terms

        Returns:
            Analysis results
        """
        print(f"📊 Analyzing document...")

        results = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "paragraph_count": len(text.split('\n\n')),
        }

        # Extract entities
        if extract_entities:
            print(f"   Extracting entities...")
            entities = self.entity_extractor.extract_from_text(text)
            results["entities"] = {
                k: [e.to_dict() for e in v]
                for k, v in entities.items()
            }
            results["entity_summary"] = self.entity_extractor.summarize_entities(entities)

        # Extract key terms
        if extract_key_terms:
            print(f"   Extracting key terms...")
            key_terms = self.entity_extractor.extract_key_terms(text, top_k=20)
            results["key_terms"] = [
                {"term": term, "frequency": freq}
                for term, freq in key_terms
            ]

        print(f"✅ Analysis complete")

        return results


# Convenience functions
def extract_entities(text: str) -> Dict[str, List[Entity]]:
    """Quick entity extraction"""
    extractor = EntityExtractor()
    return extractor.extract_from_text(text)


def analyze_document(text: str) -> Dict[str, Any]:
    """Quick document analysis"""
    analyzer = DocumentAnalyzer()
    return analyzer.analyze_document(text)


if __name__ == "__main__":
    # Test entity extraction
    print("Entity Extraction Test")
    print("=" * 50)

    test_text = """
John Smith works at Microsoft Corporation in New York.
He can be reached at john.smith@microsoft.com or 555-123-4567.
The meeting is scheduled for 12/25/2024 at 3:00 PM.
The project budget is $50,000 with a 15% contingency.
Visit https://microsoft.com for more information.
    """

    extractor = EntityExtractor(use_llm=False)
    entities = extractor.extract_from_text(test_text)

    print(f"\nExtracted Entities:")
    for entity_type, entity_list in entities.items():
        print(f"\n{entity_type.upper()}:")
        for entity in entity_list:
            print(f"  - {entity.text} (found {entity.occurrences} times)")

    # Test key terms
    key_terms = extractor.extract_key_terms(test_text, top_k=5)
    print(f"\nKey Terms:")
    for term, freq in key_terms:
        print(f"  - {term}: {freq}")
