from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import config
from logger import get_logger
from cache_manager import cached

logger = get_logger(__name__)

class GraphDBService:
    """Service for interacting with Neo4j graph database."""
    
    def __init__(
        self,
        uri: str = config.NEO4J_URI,
        user: str = config.NEO4J_USER,
        password: str = config.NEO4J_PASSWORD
    ):
        """
        Initialize Neo4j service.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
    
    def create_constraints(self):
        """Create database constraints."""
        with self.driver.session() as session:
            try:
                session.run(
                    "CREATE CONSTRAINT IF NOT EXISTS "
                    "FOR (n:Entity) REQUIRE n.id IS UNIQUE"
                )
                logger.info("Created uniqueness constraint on Entity.id")
            except Exception as e:
                logger.error(f"Error creating constraints: {e}")
                raise
    
    def upsert_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upsert a node in Neo4j.
        
        Args:
            node: Node dictionary with properties
            
        Returns:
            Result dictionary
        """
        with self.driver.session() as session:
            try:
                # Extract labels
                node_type = node.get("type", "Unknown")
                labels = [node_type, "Entity"]
                label_cypher = ":" + ":".join(labels)
                
                # Remove connections from properties
                props = {k: v for k, v in node.items() if k != "connections"}
                
                result = session.run(
                    f"MERGE (n{label_cypher} {{id: $id}}) "
                    "SET n += $props "
                    "RETURN n.id as id",
                    id=node["id"],
                    props=props
                )
                
                record = result.single()
                logger.debug(f"Upserted node: {record['id']}")
                return {"id": record["id"], "status": "success"}
            except Exception as e:
                logger.error(f"Error upserting node {node.get('id')}: {e}")
                raise
    
    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Create a relationship between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            rel_type: Relationship type
            properties: Optional relationship properties
            
        Returns:
            Result dictionary
        """
        with self.driver.session() as session:
            try:
                props = properties or {}
                result = session.run(
                    "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    "SET r += $props "
                    "RETURN type(r) as rel_type",
                    source_id=source_id,
                    target_id=target_id,
                    props=props
                )
                
                record = result.single()
                if record:
                    logger.debug(f"Created relationship: {source_id}-[{rel_type}]->{target_id}")
                    return {"status": "success", "rel_type": record["rel_type"]}
                else:
                    logger.warning(f"Could not create relationship: nodes not found")
                    return {"status": "failed", "reason": "nodes_not_found"}
            except Exception as e:
                logger.error(f"Error creating relationship: {e}")
                raise
    
    @cached(prefix="graph_context", ttl=config.CACHE_TTL_GRAPH_CONTEXT)
    def fetch_neighborhood(
        self,
        node_id: str,
        depth: int = 1,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch neighboring nodes and relationships.
        
        Args:
            node_id: Central node ID
            depth: Traversal depth (currently only supports 1)
            limit: Maximum number of neighbors
            
        Returns:
            List of relationship facts
        """
        with self.driver.session() as session:
            try:
                query = (
                    "MATCH (n:Entity {id: $node_id})-[r]-(m:Entity) "
                    "RETURN type(r) AS rel, labels(m) AS labels, "
                    "m.id AS id, m.name AS name, m.type AS type, "
                    "m.description AS description "
                    "LIMIT $limit"
                )
                
                result = session.run(query, node_id=node_id, limit=limit)
                
                facts = []
                for record in result:
                    facts.append({
                        "source": node_id,
                        "rel": record["rel"],
                        "target_id": record["id"],
                        "target_name": record["name"],
                        "target_type": record["type"],
                        "target_desc": (record["description"] or "")[:400],
                        "labels": record["labels"]
                    })
                
                logger.debug(f"Found {len(facts)} neighbors for {node_id}")
                return facts
            except Exception as e:
                logger.error(f"Error fetching neighborhood for {node_id}: {e}")
                return []
    
    def fetch_multi_neighborhood(
        self,
        node_ids: List[str],
        depth: int = 1,
        limit_per_node: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch neighborhoods for multiple nodes.
        
        Args:
            node_ids: List of node IDs
            depth: Traversal depth
            limit_per_node: Maximum neighbors per node
            
        Returns:
            Combined list of relationship facts
        """
        all_facts = []
        for node_id in node_ids:
            facts = self.fetch_neighborhood(node_id, depth, limit_per_node)
            all_facts.extend(facts)
        
        logger.info(f"Fetched {len(all_facts)} total facts for {len(node_ids)} nodes")
        return all_facts
    
    def execute_cypher(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a custom Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        with self.driver.session() as session:
            try:
                result = session.run(query, parameters or {})
                records = [dict(record) for record in result]
                logger.debug(f"Executed query, returned {len(records)} records")
                return records
            except Exception as e:
                logger.error(f"Error executing Cypher query: {e}")
                raise
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node by ID."""
        with self.driver.session() as session:
            try:
                result = session.run(
                    "MATCH (n:Entity {id: $node_id}) RETURN n",
                    node_id=node_id
                )
                record = result.single()
                if record:
                    return dict(record["n"])
                return None
            except Exception as e:
                logger.error(f"Error fetching node {node_id}: {e}")
                return None


# Global instance
graph_db_service = GraphDBService()