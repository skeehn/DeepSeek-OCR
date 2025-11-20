"""
Query Router
Intelligently routes queries to the appropriate LLM (Ollama or Gemini)
"""
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from backend.llm.ollama_client import OllamaClient
from backend.llm.gemini_client import GeminiClient, RateLimitedGeminiClient


class LLMType(Enum):
    """LLM type enum"""
    OLLAMA = "ollama"
    GEMINI = "gemini"
    AUTO = "auto"


@dataclass
class RoutingDecision:
    """Represents a routing decision"""
    llm_type: LLMType
    reason: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "llm_type": self.llm_type.value,
            "reason": self.reason,
            "confidence": self.confidence,
        }


class QueryRouter:
    """
    Routes queries to appropriate LLM based on various factors
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        gemini_client: Optional[GeminiClient] = None,
        default_llm: LLMType = LLMType.OLLAMA
    ):
        """
        Initialize query router

        Args:
            ollama_client: Ollama client instance
            gemini_client: Gemini client instance
            default_llm: Default LLM to use
        """
        self.ollama = ollama_client
        self.gemini = gemini_client
        self.default_llm = default_llm

        # Privacy-sensitive keywords (use local)
        self.privacy_keywords = [
            'ssn', 'social security', 'medical', 'health', 'confidential',
            'private', 'personal', 'password', 'secret', 'financial',
            'credit card', 'bank account', 'salary', 'income'
        ]

        # Complex reasoning keywords (use Gemini)
        self.complex_keywords = [
            'compare', 'analyze', 'evaluate', 'assess', 'contrast',
            'explain why', 'reasoning', 'justify', 'critique',
            'synthesize', 'comprehensive', 'detailed analysis'
        ]

        # Simple query patterns (use local)
        self.simple_patterns = [
            r'^what is\s+',
            r'^who is\s+',
            r'^when was\s+',
            r'^where is\s+',
            r'^define\s+',
            r'^list\s+',
        ]

        print(f"✅ Query router initialized")
        print(f"   Default LLM: {default_llm.value}")
        print(f"   Ollama available: {self.ollama is not None}")
        print(f"   Gemini available: {self.gemini is not None}")

    def route(
        self,
        query: str,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Route query to appropriate LLM

        Args:
            query: User query
            context: Retrieved context
            metadata: Additional metadata

        Returns:
            RoutingDecision object
        """
        query_lower = query.lower()

        # Priority 1: Privacy check
        if self._contains_privacy_keywords(query_lower):
            return RoutingDecision(
                llm_type=LLMType.OLLAMA,
                reason="Privacy-sensitive content detected",
                confidence=1.0
            )

        # Priority 2: Complex reasoning
        if self._is_complex_query(query_lower):
            if self.gemini:
                return RoutingDecision(
                    llm_type=LLMType.GEMINI,
                    reason="Complex reasoning required",
                    confidence=0.9
                )

        # Priority 3: Simple factual query
        if self._is_simple_query(query_lower):
            return RoutingDecision(
                llm_type=LLMType.OLLAMA,
                reason="Simple factual query",
                confidence=0.8
            )

        # Priority 4: Long context (prefer Gemini for large context windows)
        if context and len(context) > 3000:
            if self.gemini:
                return RoutingDecision(
                    llm_type=LLMType.GEMINI,
                    reason="Large context window needed",
                    confidence=0.7
                )

        # Priority 5: Metadata hints
        if metadata:
            doc_type = metadata.get('doc_type', '')
            if doc_type in ['medical', 'legal', 'financial']:
                return RoutingDecision(
                    llm_type=LLMType.OLLAMA,
                    reason=f"Sensitive document type: {doc_type}",
                    confidence=0.9
                )

        # Default: Use Ollama for cost-effectiveness
        return RoutingDecision(
            llm_type=self.default_llm,
            reason="Default routing (cost-effective)",
            confidence=0.6
        )

    def _contains_privacy_keywords(self, query: str) -> bool:
        """Check if query contains privacy-sensitive keywords"""
        for keyword in self.privacy_keywords:
            if keyword in query:
                return True
        return False

    def _is_complex_query(self, query: str) -> bool:
        """Check if query requires complex reasoning"""
        for keyword in self.complex_keywords:
            if keyword in query:
                return True

        # Check for multiple questions (indicates complexity)
        question_marks = query.count('?')
        if question_marks > 1:
            return True

        # Check query length (longer queries often more complex)
        if len(query.split()) > 20:
            return True

        return False

    def _is_simple_query(self, query: str) -> bool:
        """Check if query is simple factual"""
        for pattern in self.simple_patterns:
            if re.match(pattern, query, re.IGNORECASE):
                return True
        return False

    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        system: Optional[str] = None,
        llm_type: LLMType = LLMType.AUTO,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate response using routed LLM

        Args:
            query: User query
            context: Retrieved context
            system: System message
            llm_type: Force specific LLM (or AUTO for routing)
            temperature: Sampling temperature

        Returns:
            Response dictionary with text and metadata
        """
        # Build full prompt
        if context:
            full_prompt = f"{context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            full_prompt = query

        # Route if AUTO
        if llm_type == LLMType.AUTO:
            routing = self.route(query, context)
            llm_type = routing.llm_type
            routing_info = routing.to_dict()
        else:
            routing_info = {
                "llm_type": llm_type.value,
                "reason": "User specified",
                "confidence": 1.0
            }

        print(f"🔀 Routing to: {llm_type.value}")
        print(f"   Reason: {routing_info['reason']}")

        # Generate with selected LLM
        if llm_type == LLMType.OLLAMA:
            if not self.ollama:
                return {
                    "text": "Error: Ollama not available",
                    "llm_type": "none",
                    "error": "Ollama client not initialized"
                }

            response = self.ollama.generate(
                full_prompt,
                system=system,
                temperature=temperature
            )

            return {
                "text": response.text,
                "llm_type": "ollama",
                "model": response.model,
                "tokens": response.total_tokens,
                "routing": routing_info
            }

        elif llm_type == LLMType.GEMINI:
            if not self.gemini:
                return {
                    "text": "Error: Gemini not available",
                    "llm_type": "none",
                    "error": "Gemini client not initialized"
                }

            response = self.gemini.generate(
                full_prompt,
                system=system,
                temperature=temperature
            )

            return {
                "text": response.text,
                "llm_type": "gemini",
                "model": response.model,
                "tokens": response.total_tokens,
                "routing": routing_info
            }

        else:
            return {
                "text": "Error: Unknown LLM type",
                "llm_type": "none",
                "error": f"Unknown LLM type: {llm_type}"
            }


class DualLLMManager:
    """
    High-level manager for dual LLM system
    Handles fallback and load balancing
    """

    def __init__(
        self,
        ollama_available: bool = True,
        gemini_available: bool = True,
        prefer_local: bool = True
    ):
        """
        Initialize dual LLM manager

        Args:
            ollama_available: Enable Ollama
            gemini_available: Enable Gemini
            prefer_local: Prefer local (Ollama) when possible
        """
        self.ollama_client = None
        self.gemini_client = None

        # Initialize clients
        if ollama_available:
            try:
                self.ollama_client = OllamaClient()
                if not self.ollama_client.is_available():
                    print("⚠️ Ollama server not running")
                    self.ollama_client = None
            except Exception as e:
                print(f"⚠️ Could not initialize Ollama: {e}")

        if gemini_available:
            try:
                self.gemini_client = RateLimitedGeminiClient()
            except Exception as e:
                print(f"⚠️ Could not initialize Gemini: {e}")

        # Create router
        default_llm = LLMType.OLLAMA if prefer_local else LLMType.GEMINI

        self.router = QueryRouter(
            ollama_client=self.ollama_client,
            gemini_client=self.gemini_client,
            default_llm=default_llm
        )

        print(f"✅ Dual LLM Manager initialized")
        print(f"   Prefer local: {prefer_local}")

    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        llm_type: LLMType = LLMType.AUTO,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response with automatic fallback

        Args:
            query: User query
            context: Context from RAG
            llm_type: LLM type (AUTO, OLLAMA, GEMINI)
            **kwargs: Additional arguments

        Returns:
            Response dictionary
        """
        # Try primary LLM
        response = self.router.generate(
            query=query,
            context=context,
            llm_type=llm_type,
            **kwargs
        )

        # If error, try fallback
        if "error" in response:
            print(f"⚠️ Primary LLM failed, trying fallback...")

            # Try opposite LLM
            if response.get("llm_type") == "ollama" and self.gemini_client:
                fallback_response = self.router.generate(
                    query=query,
                    context=context,
                    llm_type=LLMType.GEMINI,
                    **kwargs
                )
                if "error" not in fallback_response:
                    fallback_response["fallback"] = True
                    return fallback_response

            elif response.get("llm_type") == "gemini" and self.ollama_client:
                fallback_response = self.router.generate(
                    query=query,
                    context=context,
                    llm_type=LLMType.OLLAMA,
                    **kwargs
                )
                if "error" not in fallback_response:
                    fallback_response["fallback"] = True
                    return fallback_response

        return response

    def is_available(self) -> bool:
        """Check if at least one LLM is available"""
        return self.ollama_client is not None or self.gemini_client is not None


# Convenience functions
def create_dual_llm_manager(prefer_local: bool = True) -> DualLLMManager:
    """Create dual LLM manager"""
    return DualLLMManager(prefer_local=prefer_local)


if __name__ == "__main__":
    # Test query router
    print("Query Router Test")
    print("=" * 50)

    manager = DualLLMManager(prefer_local=True)

    if manager.is_available():
        # Test queries
        test_queries = [
            "What is machine learning?",
            "Compare supervised and unsupervised learning approaches in detail.",
            "What is my SSN?",
            "Analyze the implications of this medical diagnosis.",
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            routing = manager.router.route(query)
            print(f"  Routed to: {routing.llm_type.value}")
            print(f"  Reason: {routing.reason}")
            print(f"  Confidence: {routing.confidence:.2f}")
    else:
        print("❌ No LLM available")
