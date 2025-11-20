"""
RAG (Retrieval-Augmented Generation) Engine
Orchestrates document retrieval and prepares for LLM response generation
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from backend.vectordb.retrieval import DocumentRetriever, RetrievalResult
from backend.vectordb.chroma_manager import ChromaManager


@dataclass
class RAGQuery:
    """Represents a RAG query"""
    query: str
    collection: str = "documents"
    top_k: int = 5
    filter_doc_id: Optional[str] = None
    include_sources: bool = True


@dataclass
class RAGResponse:
    """Represents a RAG response"""
    query: str
    answer: str
    sources: List[RetrievalResult]
    context: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "context": self.context,
            "metadata": self.metadata,
        }


class RAGEngine:
    """
    RAG Engine for question answering over documents
    Retrieves relevant context and prepares for LLM generation
    """

    def __init__(
        self,
        collection_name: str = "documents",
        chroma_manager: Optional[ChromaManager] = None
    ):
        """
        Initialize RAG engine

        Args:
            collection_name: ChromaDB collection name
            chroma_manager: ChromaDB manager instance
        """
        self.collection_name = collection_name

        # Initialize retriever
        self.retriever = DocumentRetriever(
            collection_name=collection_name,
            chroma_manager=chroma_manager
        )

        print(f"✅ RAG Engine initialized")
        print(f"   Collection: {collection_name}")

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_doc_id: Optional[str] = None,
        max_context_length: int = 2000
    ) -> Dict[str, Any]:
        """
        Execute a RAG query

        Args:
            query_text: User query
            top_k: Number of chunks to retrieve
            filter_doc_id: Optional document ID filter
            max_context_length: Maximum context length

        Returns:
            Query results dictionary
        """
        print(f"\n🔍 RAG Query: {query_text}")

        # Step 1: Retrieve relevant chunks
        print(f"   Retrieving top {top_k} chunks...")
        results = self.retriever.search(
            query=query_text,
            top_k=top_k,
            filter_doc_id=filter_doc_id
        )

        # Step 2: Build context
        context = self._build_context(results, max_context_length)

        # Step 3: Prepare metadata
        metadata = {
            "num_chunks_retrieved": len(results),
            "context_length": len(context),
            "collection": self.collection_name,
            "filter_doc_id": filter_doc_id,
        }

        print(f"   Retrieved: {len(results)} chunks")
        print(f"   Context length: {len(context)} characters")

        return {
            "query": query_text,
            "context": context,
            "sources": results,
            "metadata": metadata,
            "prompt_ready": True,  # Ready for LLM
        }

    def build_prompt(
        self,
        query: str,
        context: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        Build a prompt for LLM with retrieved context

        Args:
            query: User query
            context: Retrieved context
            system_message: Optional system message

        Returns:
            Formatted prompt string
        """
        default_system = """You are a helpful AI assistant that answers questions based on the provided context.
Use only the information from the context to answer the question.
If the context doesn't contain enough information, say so clearly.
Provide specific references to the source documents when possible."""

        system = system_message or default_system

        prompt = f"""{system}

Context from documents:
{context}

Question: {query}

Answer:"""

        return prompt

    def answer_query(
        self,
        query_text: str,
        top_k: int = 5,
        use_llm: bool = False,
        llm_handler: Optional[callable] = None
    ) -> RAGResponse:
        """
        Answer a query using RAG

        Args:
            query_text: User query
            top_k: Number of chunks to retrieve
            use_llm: Whether to use LLM for generation
            llm_handler: Optional LLM callable (takes prompt, returns answer)

        Returns:
            RAGResponse object
        """
        # Get context
        query_result = self.query(query_text, top_k=top_k)

        context = query_result["context"]
        sources = query_result["sources"]

        # Generate answer
        if use_llm and llm_handler:
            # Build prompt
            prompt = self.build_prompt(query_text, context)

            # Call LLM
            answer = llm_handler(prompt)
        else:
            # Return context without LLM generation
            answer = f"Found {len(sources)} relevant passages. LLM integration pending."

        # Create response
        response = RAGResponse(
            query=query_text,
            answer=answer,
            sources=sources,
            context=context,
            metadata=query_result["metadata"]
        )

        return response

    def multi_turn_query(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[RAGResponse]:
        """
        Handle multiple related queries

        Args:
            queries: List of user queries
            top_k: Number of chunks per query

        Returns:
            List of RAG responses
        """
        responses = []

        for query_text in queries:
            response = self.answer_query(query_text, top_k=top_k)
            responses.append(response)

        return responses

    def get_document_summary(
        self,
        doc_id: str,
        max_chunks: int = 10
    ) -> Dict[str, Any]:
        """
        Get a summary view of a document's chunks

        Args:
            doc_id: Document ID
            max_chunks: Maximum chunks to retrieve

        Returns:
            Document summary dictionary
        """
        print(f"\n📄 Getting summary for document: {doc_id}")

        # Get chunks for document
        chunks = self.retriever.get_document_chunks(doc_id, limit=max_chunks)

        # Build summary
        summary = {
            "doc_id": doc_id,
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text_preview": chunk.text[:100] + "...",
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
        }

        return summary

    def compare_documents(
        self,
        query: str,
        doc_ids: List[str],
        top_k_per_doc: int = 3
    ) -> Dict[str, Any]:
        """
        Compare how multiple documents address a query

        Args:
            query: Query to search for
            doc_ids: List of document IDs to compare
            top_k_per_doc: Results per document

        Returns:
            Comparison results
        """
        print(f"\n🔄 Comparing {len(doc_ids)} documents")
        print(f"   Query: {query}")

        comparison = {
            "query": query,
            "documents": {},
        }

        for doc_id in doc_ids:
            results = self.retriever.search(
                query=query,
                top_k=top_k_per_doc,
                filter_doc_id=doc_id
            )

            comparison["documents"][doc_id] = {
                "num_results": len(results),
                "results": [r.to_dict() for r in results],
                "best_score": results[0].score if results else 0.0,
            }

        return comparison

    def _build_context(
        self,
        results: List[RetrievalResult],
        max_length: int = 2000
    ) -> str:
        """
        Build context string from retrieval results

        Args:
            results: List of retrieval results
            max_length: Maximum context length

        Returns:
            Context string
        """
        context_parts = []
        total_length = 0

        for i, result in enumerate(results):
            # Create source reference
            source_ref = f"[Source {i+1}: Doc {result.doc_id}, Chunk {result.chunk_id}, Score: {result.score:.3f}]"

            chunk_text = result.text
            chunk_length = len(chunk_text)

            # Check length
            if total_length + chunk_length + len(source_ref) > max_length:
                break

            context_parts.append(source_ref)
            context_parts.append(chunk_text)
            context_parts.append("")  # Separator

            total_length += chunk_length + len(source_ref)

        return "\n".join(context_parts)


class ConversationalRAG:
    """
    RAG engine with conversation history support
    """

    def __init__(
        self,
        collection_name: str = "documents",
        max_history: int = 5
    ):
        """
        Initialize conversational RAG

        Args:
            collection_name: ChromaDB collection
            max_history: Maximum conversation history to keep
        """
        self.rag_engine = RAGEngine(collection_name=collection_name)
        self.conversation_history = []
        self.max_history = max_history

        print(f"✅ Conversational RAG initialized")
        print(f"   Max history: {max_history} turns")

    def chat(
        self,
        query: str,
        top_k: int = 5,
        use_llm: bool = False,
        llm_handler: Optional[callable] = None
    ) -> RAGResponse:
        """
        Chat with the RAG system

        Args:
            query: User query
            top_k: Number of chunks
            use_llm: Use LLM
            llm_handler: LLM callable

        Returns:
            RAG response
        """
        # Get response
        response = self.rag_engine.answer_query(
            query_text=query,
            top_k=top_k,
            use_llm=use_llm,
            llm_handler=llm_handler
        )

        # Add to history
        self.conversation_history.append({
            "query": query,
            "response": response.answer,
        })

        # Trim history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        return response

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared")


# Convenience functions
def create_rag_engine(collection_name: str = "documents") -> RAGEngine:
    """Create RAG engine"""
    return RAGEngine(collection_name=collection_name)


def quick_rag_query(query: str, collection: str = "documents", top_k: int = 5) -> Dict[str, Any]:
    """
    Quick RAG query

    Args:
        query: User query
        collection: Collection name
        top_k: Number of results

    Returns:
        Query results
    """
    engine = RAGEngine(collection_name=collection)
    return engine.query(query, top_k=top_k)


if __name__ == "__main__":
    # Test RAG engine
    print("RAG Engine Test")
    print("=" * 50)

    # Initialize
    engine = RAGEngine(collection_name="test_documents")

    # Test query
    query_result = engine.query(
        query_text="What is machine learning?",
        top_k=3
    )

    print(f"\nQuery results:")
    print(f"  Query: {query_result['query']}")
    print(f"  Sources: {query_result['metadata']['num_chunks_retrieved']}")
    print(f"  Context length: {query_result['metadata']['context_length']}")

    # Test prompt building
    prompt = engine.build_prompt(
        query="What is deep learning?",
        context="Deep learning is a subset of machine learning..."
    )

    print(f"\nGenerated prompt:")
    print(prompt[:200] + "...")

    # Test conversational RAG
    conv_rag = ConversationalRAG()
    print(f"\n✅ Conversational RAG ready for chat")
