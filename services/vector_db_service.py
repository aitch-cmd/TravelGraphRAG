from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
import config
from logger import get_logger
from cache_manager import cached
from services.embedding_service import embedding_service

logger = get_logger(__name__)

class VectorDBService:
    """Service for interacting with Pinecone vector database."""
    
    def __init__(
        self,
        index_name: str = config.PINECONE_INDEX_NAME,
        dimension: int = config.PINECONE_VECTOR_DIM
    ):
        """
        Initialize Pinecone service.
        
        Args:
            index_name: Name of the Pinecone index
            dimension: Vector dimension
        """
        self.index_name = index_name
        self.dimension = dimension
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        
        # Create or connect to index
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        
        logger.info(f"Initialized VectorDBService with index: {index_name}")
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        existing_indexes = self.pc.list_indexes().names()
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Created index: {self.index_name}")
        else:
            logger.info(f"Using existing index: {self.index_name}")
    
    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Upsert vectors to Pinecone.
        
        Args:
            vectors: List of vector dicts with 'id', 'values', 'metadata'
            batch_size: Batch size for upsert
            
        Returns:
            Dict with upsert statistics
        """
        if not vectors:
            logger.warning("No vectors to upsert")
            return {"upserted": 0}
        
        total_upserted = 0
        
        # Batch upsert
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                result = self.index.upsert(vectors=batch)
                upserted = result.get('upserted_count', len(batch))
                total_upserted += upserted
                logger.debug(f"Upserted batch {i//batch_size + 1}: {upserted} vectors")
            except Exception as e:
                logger.error(f"Error upserting batch {i//batch_size + 1}: {e}")
                raise
        
        logger.info(f"Total upserted: {total_upserted} vectors")
        return {"upserted": total_upserted}
    
    @cached(prefix="vector_search", ttl=config.CACHE_TTL_SEARCH_RESULTS)
    def search(
        self,
        query_text: str,
        top_k: int = config.TOP_K,
        filter_dict: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using text query.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            filter_dict: Metadata filter
            include_metadata: Whether to include metadata
            
        Returns:
            List of matches with id, score, and metadata
        """
        try:
            # Generate query embedding
            logger.debug(f"Searching for: {query_text[:50]}...")
            query_vector = embedding_service.embed_single(query_text)
            
            # Search Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                include_values=False,
                filter=filter_dict
            )
            
            matches = results.get("matches", [])
            logger.info(f"Found {len(matches)} matches for query")
            
            return [
                {
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match.get("metadata", {})
                }
                for match in matches
            ]
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise
    
    def search_by_vector(
        self,
        query_vector: List[float],
        top_k: int = config.TOP_K,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search using a pre-computed vector."""
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                include_values=False,
                filter=filter_dict
            )
            
            matches = results.get("matches", [])
            logger.info(f"Found {len(matches)} matches")
            
            return [
                {
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match.get("metadata", {})
                }
                for match in matches
            ]
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise
    
    def delete_vectors(self, ids: List[str]) -> Dict[str, Any]:
        """Delete vectors by IDs."""
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors")
            return {"deleted": len(ids)}
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            raise


# Global instance
vector_db_service = VectorDBService()