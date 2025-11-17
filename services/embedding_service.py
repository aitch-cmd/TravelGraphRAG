from typing import List, Union
from openai import OpenAI
import config
from logger import get_logger
from cache_manager import cached, cache_manager

logger = get_logger(__name__)

class EmbeddingService:
    """Service for generating text embeddings with caching."""

    def __init__(self, model: str = config.EMBED_MODEL):
        """
        Initialize embedding service.
        
        Args:
            model: OpenAI embedding model name
        """
        self.model = model
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        logger.info(f"Initialized EmbeddingService with model: {model}")

    def _generate_cache_key(self, text: Union[str, List[str]]) -> str:
        """Generate cache key for embeddings."""
        if isinstance(text, list):
            text = "|".join(text[:3])  
        return f"embedding:{self.model}:{hash(text)}"
        
    @cached(prefix="embedding", ttl=config.CACHE_TTL_EMBEDDINGS)
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            logger.debug(f"Generating embedding for text: {text[:50]}...")
            response = self.client.embeddings.create(
                model=self.model,
                input=[text]
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding with dimension: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise  
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with caching.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("Empty text list provided")
            return []
        
        embeddings = []
        cache_keys = [self._generate_cache_key(text) for text in texts]
        
        # Check cache for each text
        to_compute = []
        to_compute_indices = []
        
        for i, (text, cache_key) in enumerate(zip(texts, cache_keys)):
            cached_embedding = cache_manager.get(cache_key)
            if cached_embedding:
                embeddings.append(cached_embedding)
                logger.debug(f"Using cached embedding for text {i+1}/{len(texts)}")
            else:
                embeddings.append(None)
                to_compute.append(text)
                to_compute_indices.append(i)
        
        # Compute missing embeddings
        if to_compute:
            logger.info(f"Computing {len(to_compute)} new embeddings")
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=to_compute
                )
                
                # Update results and cache
                for idx, data in zip(to_compute_indices, response.data):
                    embedding = data.embedding
                    embeddings[idx] = embedding
                    
                    # Cache the result
                    cache_key = cache_keys[idx]
                    cache_manager.set(
                        cache_key,
                        embedding,
                        ttl=config.CACHE_TTL_EMBEDDINGS
                    )
                
                logger.info(f"Successfully computed and cached {len(to_compute)} embeddings")
            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                raise
        
        return embeddings
    
    def clear_cache(self):
        """Clear all embedding caches."""
        logger.info("Clearing embedding cache")
        cache_manager.clear_pattern(f"embedding:{self.model}:*")


# Global instance
embedding_service = EmbeddingService()