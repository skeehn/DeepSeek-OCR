"""
Complete RAG Pipeline with LLM Integration
Combines retrieval, routing, and generation for end-to-end RAG
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from backend.features.rag_engine import RAGEngine, RAGResponse
from backend.llm.query_router import DualLLMManager, LLMType
from backend.vectordb.retrieval import DocumentRetriever


@dataclass
class CompletePipelineResponse:
    """Complete RAG pipeline response"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    context: str
    llm_used: str
    tokens_used: int
    routing_reason: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "context_preview": self.context[:200] + "..." if len(self.context) > 200 else self.context,
            "llm_used": self.llm_used,
            "tokens_used": self.tokens_used,
            "routing_reason": self.routing_reason,
            "metadata": self.metadata,
        }


class CompleteRAGPipeline:
    """
    Complete RAG pipeline with automatic LLM routing
    End-to-end: Query → Retrieval → Routing → Generation → Response
    """

    def __init__(
        self,
        collection_name: str = "documents",
        prefer_local: bool = True,
        enable_ollama: bool = True,
        enable_gemini: bool = True
    ):
        """
        Initialize complete RAG pipeline

        Args:
            collection_name: ChromaDB collection
            prefer_local: Prefer local LLM (Ollama)
            enable_ollama: Enable Ollama
            enable_gemini: Enable Gemini
        """
        self.collection_name = collection_name

        # Initialize RAG engine
        self.rag_engine = RAGEngine(collection_name=collection_name)

        # Initialize dual LLM manager
        self.llm_manager = DualLLMManager(
            ollama_available=enable_ollama,
            gemini_available=enable_gemini,
            prefer_local=prefer_local
        )

        if not self.llm_manager.is_available():
            print("⚠️ Warning: No LLM available")

        print(f"✅ Complete RAG Pipeline initialized")
        print(f"   Collection: {collection_name}")
        print(f"   Prefer local: {prefer_local}")

    def query(
        self,
        query: str,
        top_k: int = 5,
        llm_type: LLMType = LLMType.AUTO,
        temperature: float = 0.7,
        include_sources: bool = True,
        filter_doc_id: Optional[str] = None
    ) -> CompleteRAGPipelineResponse:
        """
        Complete RAG query with answer generation

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            llm_type: LLM to use (AUTO, OLLAMA, GEMINI)
            temperature: Sampling temperature
            include_sources: Include source chunks in response
            filter_doc_id: Filter by specific document

        Returns:
            CompleteRAGPipelineResponse
        """
        print(f"\n🔍 RAG Pipeline Query: {query}")

        # Step 1: Retrieve context
        print(f"   Step 1/3: Retrieving context...")
        rag_result = self.rag_engine.query(
            query_text=query,
            top_k=top_k,
            filter_doc_id=filter_doc_id
        )

        context = rag_result["context"]
        sources = rag_result["sources"]

        print(f"   Retrieved {len(sources)} relevant chunks")

        # Step 2: Build prompt
        print(f"   Step 2/3: Building prompt...")
        prompt = self.rag_engine.build_prompt(query, context)

        # Step 3: Generate answer
        print(f"   Step 3/3: Generating answer...")
        llm_response = self.llm_manager.generate(
            query=query,
            context=context,
            llm_type=llm_type,
            temperature=temperature
        )

        print(f"   ✅ Answer generated using {llm_response.get('llm_type', 'unknown')}")

        # Build response
        response = CompletePipelineResponse(
            query=query,
            answer=llm_response.get("text", "No answer generated"),
            sources=[s.to_dict() for s in sources] if include_sources else [],
            context=context,
            llm_used=llm_response.get("llm_type", "unknown"),
            tokens_used=llm_response.get("tokens", 0),
            routing_reason=llm_response.get("routing", {}).get("reason", "N/A"),
            metadata={
                "num_chunks": len(sources),
                "context_length": len(context),
                "model": llm_response.get("model", "unknown"),
                **rag_result.get("metadata", {})
            }
        )

        return response

    def ask(
        self,
        question: str,
        use_local: bool = True,
        **kwargs
    ) -> str:
        """
        Simple question answering (returns just the answer text)

        Args:
            question: Question to ask
            use_local: Use local LLM (Ollama)
            **kwargs: Additional arguments

        Returns:
            Answer text
        """
        llm_type = LLMType.OLLAMA if use_local else LLMType.GEMINI

        response = self.query(question, llm_type=llm_type, **kwargs)
        return response.answer

    def summarize_document(
        self,
        doc_id: str,
        max_chunks: int = 10,
        use_cloud: bool = True
    ) -> str:
        """
        Summarize a document

        Args:
            doc_id: Document ID
            max_chunks: Maximum chunks to use
            use_cloud: Use cloud LLM (Gemini) for better quality

        Returns:
            Document summary
        """
        print(f"\n📄 Summarizing document: {doc_id}")

        # Get document summary from RAG engine
        doc_summary = self.rag_engine.get_document_summary(doc_id, max_chunks=max_chunks)

        # Build summarization prompt
        chunks_text = "\n\n".join([
            chunk["text_preview"]
            for chunk in doc_summary["chunks"]
        ])

        prompt = f"""Please provide a comprehensive summary of the following document.

Document Content:
{chunks_text}

Summary:"""

        # Generate summary
        llm_type = LLMType.GEMINI if use_cloud else LLMType.OLLAMA

        llm_response = self.llm_manager.generate(
            query=prompt,
            llm_type=llm_type,
            temperature=0.5
        )

        return llm_response.get("text", "Could not generate summary")

    def compare_and_analyze(
        self,
        query: str,
        doc_ids: List[str],
        use_cloud: bool = True
    ) -> str:
        """
        Compare multiple documents and provide analysis

        Args:
            query: Query for comparison
            doc_ids: Document IDs to compare
            use_cloud: Use cloud LLM

        Returns:
            Comparison analysis
        """
        print(f"\n🔄 Comparing {len(doc_ids)} documents")

        # Get comparison data
        comparison = self.rag_engine.compare_documents(
            query=query,
            doc_ids=doc_ids
        )

        # Build comparison prompt
        comparison_text = f"Query: {query}\n\n"

        for doc_id, doc_data in comparison["documents"].items():
            comparison_text += f"Document: {doc_id}\n"
            comparison_text += f"Relevance Score: {doc_data['best_score']:.3f}\n"

            if doc_data["results"]:
                comparison_text += "Key passages:\n"
                for result in doc_data["results"][:2]:
                    comparison_text += f"- {result['text'][:200]}...\n"

            comparison_text += "\n"

        prompt = f"""Compare and analyze the following documents based on the query.

{comparison_text}

Please provide a comparative analysis highlighting similarities, differences, and key insights:"""

        # Generate analysis
        llm_type = LLMType.GEMINI if use_cloud else LLMType.OLLAMA

        llm_response = self.llm_manager.generate(
            query=prompt,
            llm_type=llm_type,
            temperature=0.7
        )

        return llm_response.get("text", "Could not generate analysis")


class ConversationalRAGPipeline:
    """
    Conversational RAG with LLM integration
    Maintains conversation history and context
    """

    def __init__(
        self,
        collection_name: str = "documents",
        max_history: int = 5,
        prefer_local: bool = True
    ):
        """
        Initialize conversational RAG pipeline

        Args:
            collection_name: ChromaDB collection
            max_history: Maximum conversation turns to remember
            prefer_local: Prefer local LLM
        """
        self.pipeline = CompleteRAGPipeline(
            collection_name=collection_name,
            prefer_local=prefer_local
        )

        self.conversation_history = []
        self.max_history = max_history

        print(f"✅ Conversational RAG Pipeline initialized")
        print(f"   Max history: {max_history} turns")

    def chat(
        self,
        message: str,
        top_k: int = 5,
        use_local: bool = True,
        **kwargs
    ) -> str:
        """
        Chat with the RAG system

        Args:
            message: User message
            top_k: Number of chunks to retrieve
            use_local: Use local LLM
            **kwargs: Additional arguments

        Returns:
            Assistant response
        """
        # Add conversation history to context
        history_context = self._build_history_context()

        # Build enhanced query with history
        if history_context:
            enhanced_query = f"{history_context}\n\nCurrent question: {message}"
        else:
            enhanced_query = message

        # Get response
        llm_type = LLMType.OLLAMA if use_local else LLMType.GEMINI

        response = self.pipeline.query(
            query=enhanced_query,
            top_k=top_k,
            llm_type=llm_type,
            **kwargs
        )

        # Add to history
        self.conversation_history.append({
            "user": message,
            "assistant": response.answer,
        })

        # Trim history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        return response.answer

    def _build_history_context(self) -> str:
        """Build context from conversation history"""
        if not self.conversation_history:
            return ""

        history_parts = []
        for turn in self.conversation_history[-3:]:  # Last 3 turns
            history_parts.append(f"User: {turn['user']}")
            history_parts.append(f"Assistant: {turn['assistant']}")

        return "Previous conversation:\n" + "\n".join(history_parts)

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared")


# Convenience functions
def create_rag_pipeline(
    collection_name: str = "documents",
    prefer_local: bool = True
) -> CompleteRAGPipeline:
    """Create complete RAG pipeline"""
    return CompleteRAGPipeline(
        collection_name=collection_name,
        prefer_local=prefer_local
    )


def quick_ask(
    question: str,
    collection: str = "documents",
    use_local: bool = True
) -> str:
    """
    Quick question answering

    Args:
        question: Question to ask
        collection: Collection name
        use_local: Use local LLM

    Returns:
        Answer text
    """
    pipeline = CompleteRAGPipeline(collection_name=collection)
    return pipeline.ask(question, use_local=use_local)


if __name__ == "__main__":
    # Test complete RAG pipeline
    print("Complete RAG Pipeline Test")
    print("=" * 50)

    try:
        pipeline = CompleteRAGPipeline(
            collection_name="test_documents",
            prefer_local=True
        )

        # Test query
        response = pipeline.query(
            query="What is machine learning?",
            top_k=3,
            llm_type=LLMType.AUTO
        )

        print(f"\n✅ Query successful!")
        print(f"   Question: {response.query}")
        print(f"   Answer: {response.answer[:200]}...")
        print(f"   LLM used: {response.llm_used}")
        print(f"   Tokens: {response.tokens_used}")
        print(f"   Sources: {len(response.sources)}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
