from typing import List, Dict, Any
from openai import OpenAI
import config
from logger import get_logger
from services.vector_db_service import vector_db_service
from services.graph_db_service import graph_db_service

logger = get_logger(__name__)

class HybridChatService:
    """Service for hybrid RAG chat combining vector and graph databases."""
    
    def __init__(self, chat_model: str = config.CHAT_MODEL):
        """
        Initialize hybrid chat service.
        
        Args:
            chat_model: OpenAI chat model to use
        """
        self.chat_model = chat_model
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        logger.info(f"Initialized HybridChatService with model: {chat_model}")
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = config.TOP_K,
        graph_depth: int = 1
    ) -> Dict[str, Any]:
        """
        Retrieve context from both vector and graph databases.
        
        Args:
            query: User query
            top_k: Number of vector search results
            graph_depth: Depth of graph traversal
            
        Returns:
            Dictionary with vector matches and graph facts
        """
        logger.info(f"Retrieving context for query: {query[:50]}...")
        
        # 1. Vector search
        logger.debug("Performing vector search")
        vector_matches = vector_db_service.search(
            query_text=query,
            top_k=top_k,
            include_metadata=True
        )
        logger.info(f"Retrieved {len(vector_matches)} vector matches")
        
        # 2. Extract node IDs from matches
        node_ids = [match["id"] for match in vector_matches]
        
        # 3. Graph context retrieval
        logger.debug("Fetching graph context")
        graph_facts = graph_db_service.fetch_multi_neighborhood(
            node_ids=node_ids,
            depth=graph_depth,
            limit_per_node=10
        )
        logger.info(f"Retrieved {len(graph_facts)} graph facts")
        
        return {
            "vector_matches": vector_matches,
            "graph_facts": graph_facts,
            "query": query
        }
    
    def build_prompt(
        self,
        query: str,
        vector_matches: List[Dict[str, Any]],
        graph_facts: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Build chat prompt from retrieved context.
        
        Args:
            query: User query
            vector_matches: Results from vector search
            graph_facts: Results from graph traversal
            
        Returns:
            List of messages for chat completion
        """
        # System message
        system_msg = (
            "You are a helpful travel assistant for Vietnam. "
            "Use the provided semantic search results and graph facts to answer "
            "the user's query briefly and concisely. "
            "Cite node IDs when referencing specific places or attractions."
        )
        
        # Build vector context
        vec_context = []
        for match in vector_matches:
            meta = match["metadata"]
            score = match.get("score", 0.0)
            snippet = (
                f"- ID: {match['id']}, "
                f"Name: {meta.get('name', 'N/A')}, "
                f"Type: {meta.get('type', 'N/A')}, "
                f"Score: {score:.3f}"
            )
            if meta.get("city"):
                snippet += f", City: {meta.get('city')}"
            vec_context.append(snippet)
        
        # Build graph context
        graph_context = [
            f"- ({fact['source']}) -[{fact['rel']}]-> "
            f"({fact['target_id']}) {fact['target_name']}: "
            f"{fact['target_desc']}"
            for fact in graph_facts
        ]
        
        # User message with context
        user_content = (
            f"User query: {query}\n\n"
            "Top semantic matches (from vector DB):\n"
            f"{chr(10).join(vec_context[:10])}\n\n"
            "Graph facts (neighboring relations):\n"
            f"{chr(10).join(graph_context[:20])}\n\n"
            "Based on the above context, answer the user's question. "
            "If helpful, suggest 2-3 concrete itinerary steps or tips and "
            "mention node IDs for references."
        )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content}
        ]
        
        logger.debug(f"Built prompt with {len(vec_context)} vector matches and {len(graph_context)} graph facts")
        return messages
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 600,
        temperature: float = 0.2
    ) -> str:
        """
        Generate chat response using OpenAI API.
        
        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        try:
            logger.debug("Generating chat response")
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Generated response ({len(answer)} chars)")
            return answer
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def chat(self, query: str, top_k: int = config.TOP_K) -> Dict[str, Any]:
        """
        Complete chat flow: retrieve context, build prompt, generate response.
        
        Args:
            query: User query
            top_k: Number of vector search results
            
        Returns:
            Dictionary with query, answer, and context
        """
        try:
            # Retrieve context
            context = self.retrieve_context(query, top_k=top_k)
            
            # Build prompt
            messages = self.build_prompt(
                query=query,
                vector_matches=context["vector_matches"],
                graph_facts=context["graph_facts"]
            )
            
            # Generate response
            answer = self.generate_response(messages)
            
            return {
                "query": query,
                "answer": answer,
                "vector_matches": context["vector_matches"],
                "graph_facts": context["graph_facts"]
            }
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                "query": query,
                "answer": f"Error: {str(e)}",
                "vector_matches": [],
                "graph_facts": []
            }


def interactive_chat():
    """Run interactive chat session."""
    chat_service = HybridChatService()
    
    print("\n" + "="*60)
    print("Hybrid Travel Assistant (Vietnam)")
    print("Type 'exit' or 'quit' to end the session")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("\n🔍 Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("exit", "quit"):
                print("\n👋 Goodbye!")
                break
            
            # Get response
            result = chat_service.chat(query)
            
            # Display answer
            print("\n" + "="*60)
            print("🤖 Assistant Answer:")
            print("-"*60)
            print(result["answer"])
            print("="*60)
            
            # Display source info
            if result.get("vector_matches"):
                print(f"\n📊 Sources: {len(result['vector_matches'])} vector matches, "
                      f"{len(result['graph_facts'])} graph connections")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in interactive chat: {e}")
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    interactive_chat()