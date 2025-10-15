# load_to_neo4j.py
import json
from neo4j import GraphDatabase
from tqdm import tqdm
import config

DATA_FILE = "vietnam_travel_dataset.json"

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

def create_constraints(tx):
    """
    Define uniqueness constraint on nodes labeled 'Entity' ensuring their 'id' property is unique.
    This prevents duplicate nodes with the same ID.
    """
    # generic uniqueness constraint on id for node label Entity (we also add label specific types)
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")

def upsert_node(tx, node):
    """
    Upsert (create or update) a node in Neo4j.
    
    - The node label comes from node["type"] plus a generic 'Entity' label.
    - Properties are taken from node dictionary except 'connections' to avoid large nested objects.
    - Use MERGE on the unique 'id' property to avoid duplicates.
    - Set the node properties with the provided values.
    """
    # use label from node['type'] and always add :Entity label
    labels = [node.get("type","Unknown"), "Entity"]
    label_cypher = ":" + ":".join(labels)
    # keep a subset of properties to store (avoid storing huge nested objects)
    props = {k:v for k,v in node.items() if k not in ("connections",)}
    # set properties using parameters
    tx.run(
        f"MERGE (n{label_cypher} {{id: $id}}) "
        "SET n += $props",
        id=node["id"], props=props
    )

def create_relationship(tx, source_id, rel):
    """
    Create a relationship between two existing nodes based on their IDs.
    
    - rel is a dictionary containing 'relation' type and 'target' node ID.
    - Matches source node and target node by IDs.
    - MERGEs the relationship type between them if not existing.
    """
    # rel is like {"relation": "Located_In", "target": "city_hanoi"}
    rel_type = rel.get("relation", "RELATED_TO")
    target_id = rel.get("target")
    if not target_id:
        return
    # Create relationship if both nodes exist
    cypher = (
        "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
        f"MERGE (a)-[r:{rel_type}]->(b) "
        "RETURN r"
    )
    tx.run(cypher, source_id=source_id, target_id=target_id)

def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    with driver.session() as session:
        session.execute_write(create_constraints)
        # Upsert all nodes
        for node in tqdm(nodes, desc="Creating nodes"):
            session.execute_write(upsert_node, node)

        # Create relationships
        for node in tqdm(nodes, desc="Creating relationships"):
            conns = node.get("connections", [])
            for rel in conns:
                session.execute_write(create_relationship, node["id"], rel)

    print("Done loading into Neo4j.")

if __name__ == "__main__":
    main()
