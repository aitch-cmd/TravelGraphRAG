import json
import time
from typing import List, Dict, Any
from tqdm import tqdm
import config
from logger import get_logger
from services.embedding_service import embedding_service
from services.vector_db_service import vector_db_service
from services.graph_db_service import graph_db_service

logger = get_logger(__name__)

class DataLoader:
    """Unified data loader for both vector and graph databases."""
    
    def __init__(self, data_file: str = "vietnam_travel_dataset.json"):
        """
        Initialize data loader.
        
        Args:
            data_file: Path to JSON data file
        """
        self.data_file = data_file
        logger.info(f"Initialized DataLoader with file: {data_file}")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                nodes = json.load(f)
            logger.info(f"Loaded {len(nodes)} nodes from {self.data_file}")
            return nodes
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_to_neo4j(self, nodes: List[Dict[str, Any]]):
        """
        Load nodes and relationships into Neo4j.
        
        Args:
            nodes: List of node dictionaries
        """
        logger.info("Loading data to Neo4j...")
        
        # Create constraints
        graph_db_service.create_constraints()
        
        # Load nodes
        logger.info("Creating nodes...")
        for node in tqdm(nodes, desc="Creating nodes"):
            try:
                graph_db_service.upsert_node(node)
            except Exception as e:
                logger.error(f"Error creating node {node.get('id')}: {e}")
        
        # Create relationships
        logger.info("Creating relationships...")
        for node in tqdm(nodes, desc="Creating relationships"):
            connections = node.get("connections", [])
            for conn in connections:
                try:
                    graph_db_service.create_relationship(
                        source_id=node["id"],
                        target_id=conn.get("target"),
                        rel_type=conn.get("relation", "RELATED_TO")
                    )
                except Exception as e:
                    logger.error(f"Error creating relationship: {e}")
        
        logger.info("✅ Completed Neo4j data loading")
    
    def prepare_vectors(
        self,
        nodes: List[Dict[str, Any]],
        batch_size: int = config.BATCH_SIZE
    ) -> List[Dict[str, Any]]:
        """
        Prepare vectors for Pinecone upload.
        
        Args:
            nodes: List of node dictionaries
            batch_size: Batch size for embedding generation
            
        Returns:
            List of vector dictionaries
        """
        logger.info("Preparing vectors for Pinecone...")
        
        # Filter nodes with semantic text
        valid_nodes = []
        texts = []
        
        for node in nodes:
            semantic_text = node.get("semantic_text") or (
                node.get("description", "")[:1000]
            )
            if semantic_text.strip():
                valid_nodes.append(node)
                texts.append(semantic_text)
        
        logger.info(f"Found {len(valid_nodes)} nodes with semantic text")
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embedding_service.embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            time.sleep(0.2)  # Rate limiting
        
        # Prepare vector objects
        vectors = []
        for node, embedding in zip(valid_nodes, all_embeddings):
            metadata = {
                "id": node.get("id"),
                "type": node.get("type"),
                "name": node.get("name"),
                "city": node.get("city", node.get("region", "")),
                "tags": node.get("tags", [])
            }
            
            vectors.append({
                "id": node["id"],
                "values": embedding,
                "metadata": metadata
            })
        
        logger.info(f"Prepared {len(vectors)} vectors")
        return vectors
    
    def load_to_pinecone(
        self,
        nodes: List[Dict[str, Any]],
        batch_size: int = config.BATCH_SIZE
    ):
        """
        Load vectors to Pinecone.
        
        Args:
            nodes: List of node dictionaries
            batch_size: Batch size for processing
        """
        logger.info("Loading data to Pinecone...")
        
        # Prepare vectors
        vectors = self.prepare_vectors(nodes, batch_size)
        
        # Upload to Pinecone
        result = vector_db_service.upsert_vectors(
            vectors=vectors,
            batch_size=100
        )
        
        logger.info(f"✅ Completed Pinecone data loading: {result}")
    
    def load_all(self, batch_size: int = config.BATCH_SIZE):
        """
        Load data to both Neo4j and Pinecone.
        
        Args:
            batch_size: Batch size for processing
        """
        logger.info("="*60)
        logger.info("Starting full data load...")
        logger.info("="*60)
        
        # Load data
        nodes = self.load_data()
        
        # Load to Neo4j
        logger.info("\n📊 Step 1/2: Loading to Neo4j")
        self.load_to_neo4j(nodes)
        
        # Load to Pinecone
        logger.info("\n🔍 Step 2/2: Loading to Pinecone")
        self.load_to_pinecone(nodes, batch_size)
        
        logger.info("\n" + "="*60)
        logger.info("✅ Data loading completed successfully!")
        logger.info("="*60)


def main():
    """Main function to run data loading."""
    loader = DataLoader()
    loader.load_all()


if __name__ == "__main__":
    main()