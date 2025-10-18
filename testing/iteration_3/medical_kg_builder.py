"""
Medical Knowledge Graph Builder
Extracts entities and relationships from PDF chunks and builds Neo4j graph

Grounded in actual PDF content (neonatal medicine, pages 233-282)
"""
import sys
from pathlib import Path
import re
from typing import List, Dict, Set, Tuple

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "iteration_1"))
sys.path.append(str(Path(__file__).parent))

from config import settings
from iteration_1.opensearch_store import OpenSearchStore
from neo4j_store import Neo4jStore, Entity, Relationship


class MedicalKGBuilder:
    """
    Build medical knowledge graph from PDF chunks

    Approach:
    1. Extract entities using pattern matching (diseases, drugs, procedures, symptoms)
    2. Identify relationships from co-occurrence and patterns
    3. Populate Neo4j graph
    """

    def __init__(self, opensearch_store: OpenSearchStore, neo4j_store: Neo4jStore):
        """
        Initialize KG builder

        Args:
            opensearch_store: OpenSearch store with indexed chunks
            neo4j_store: Neo4j store for graph
        """
        self.opensearch = opensearch_store
        self.neo4j = neo4j_store

        # Medical entity patterns (from our PDF content)
        self.entity_patterns = {
            "disease": [
                "PPHN", "persistent pulmonary hypertension",
                "PDA", "patent ductus arteriosus",
                "RDS", "respiratory distress syndrome",
                "apnea of prematurity",
                "meconium aspiration",
                "hyperthyroidism",
                "hypothyroidism",
                "Graves disease",
                "sepsis",
                "pneumonia",
                "hypoglycemia",
                "hypoxia",
                "bradycardia",
                "tachycardia",
                "asphyxia",
                "neonatal HSV",
                "syphilis",
            ],
            "drug": [
                "acyclovir",
                "penicillin",
                "ampicillin",
                "nitrofurantoin",
                "propranolol",
                "oxygen",
                "surfactant",
                "ECMO",
            ],
            "procedure": [
                "cardiac massage",
                "intubation",
                "ventilation",
                "suctioning",
                "extracorporeal membrane oxygenation",
                "CPAP",
                "resuscitation",
            ],
            "symptom": [
                "respiratory distress",
                "apnea",
                "cyanosis",
                "hypoxemia",
                "tachypnea",
                "retractions",
                "grunting",
                "bradycardia",
                "edema",
            ],
            "anatomy": [
                "ductus arteriosus",
                "foramen ovale",
                "pulmonary artery",
                "umbilical cord",
                "vocal cords",
            ]
        }

        # Relationship patterns
        self.relationship_patterns = {
            "TREATS": [
                r"(\w+)\s+(?:is|for|treats?|therapy for|treatment of)\s+(\w+)",
                r"(\w+)\s+(?:administered|given)\s+(?:to|for)\s+(\w+)",
            ],
            "HAS_SYMPTOM": [
                r"(\w+)\s+(?:presents with|characterized by|symptoms include)\s+(\w+)",
                r"(\w+)\s+(?:may have|develops|shows)\s+(\w+)",
            ],
            "CAUSES": [
                r"(\w+)\s+(?:causes|results in|leads to)\s+(\w+)",
                r"(\w+)\s+(?:may cause|can result in)\s+(\w+)",
            ],
            "USED_FOR": [
                r"(\w+)\s+(?:for|in cases of|to treat)\s+(\w+)",
                r"(\w+)\s+(?:performed|used)\s+(?:for|in)\s+(\w+)",
            ],
        }

    def extract_entities_from_chunks(self, limit: int = None) -> Dict[str, Set[str]]:
        """
        Extract entities from OpenSearch chunks

        Args:
            limit: Maximum number of chunks to process (None = all)

        Returns:
            Dictionary of entity_type -> set of entity names
        """
        print(f"[INFO] Extracting entities from chunks...")

        # Get all chunks (or sample)
        # Use a broad query to get many chunks
        all_chunks = []
        for term in ["infant", "newborn", "neonatal", "treatment", "disease"]:
            results = self.opensearch.search(term, top_k=200)
            all_chunks.extend(results)

        # Remove duplicates
        unique_chunks = {c.chunk_id: c for c in all_chunks}.values()
        chunks = list(unique_chunks)[:limit] if limit else list(unique_chunks)

        print(f"[OK] Processing {len(chunks)} chunks")

        # Extract entities
        found_entities = {entity_type: set() for entity_type in self.entity_patterns}

        for chunk in chunks:
            text = chunk.text.lower()

            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in text:
                        found_entities[entity_type].add(pattern)

        # Print stats
        print(f"\n[ENTITIES FOUND]")
        for entity_type, entities in found_entities.items():
            if entities:
                print(f"  {entity_type.capitalize()}: {len(entities)}")
                for e in sorted(entities)[:5]:  # Show first 5
                    print(f"    - {e}")
                if len(entities) > 5:
                    print(f"    ... and {len(entities) - 5} more")

        return found_entities

    def extract_relationships_from_chunks(
        self,
        chunks: List,
        entities: Dict[str, Set[str]]
    ) -> List[Tuple[str, str, str]]:
        """
        Extract relationships from chunks using pattern matching

        Args:
            chunks: List of chunks
            entities: Dictionary of extracted entities

        Returns:
            List of (source, target, rel_type) tuples
        """
        print(f"\n[INFO] Extracting relationships...")

        relationships = []

        for chunk in chunks:
            text = chunk.text.lower()

            # Simple co-occurrence based relationships
            # If disease and drug appear together, likely TREATS relationship
            for disease in entities.get("disease", []):
                for drug in entities.get("drug", []):
                    if disease.lower() in text and drug.lower() in text:
                        # Check for treatment keywords
                        if any(kw in text for kw in ["treat", "therapy", "treatment", "administered"]):
                            relationships.append((drug, disease, "TREATS"))

            # Disease and symptom co-occurrence
            for disease in entities.get("disease", []):
                for symptom in entities.get("symptom", []):
                    if disease.lower() in text and symptom.lower() in text:
                        relationships.append((disease, symptom, "HAS_SYMPTOM"))

            # Procedure and disease co-occurrence
            for procedure in entities.get("procedure", []):
                for disease in entities.get("disease", []):
                    if procedure.lower() in text and disease.lower() in text:
                        relationships.append((procedure, disease, "USED_FOR"))

        # Remove duplicates
        relationships = list(set(relationships))

        print(f"[OK] Found {len(relationships)} relationships")
        for rel in relationships[:10]:  # Show first 10
            print(f"  {rel[0]} -[{rel[2]}]-> {rel[1]}")
        if len(relationships) > 10:
            print(f"  ... and {len(relationships) - 10} more")

        return relationships

    def build_graph(self, limit_chunks: int = None):
        """
        Build knowledge graph from PDF chunks

        Args:
            limit_chunks: Limit number of chunks to process (None = all)
        """
        print("="*80)
        print("BUILDING MEDICAL KNOWLEDGE GRAPH")
        print("="*80)
        print()

        # Step 1: Extract entities
        entities = self.extract_entities_from_chunks(limit=limit_chunks)

        # Step 2: Add entities to Neo4j
        print(f"\n[INFO] Adding entities to Neo4j...")
        entity_count = 0
        for entity_type, entity_names in entities.items():
            for name in entity_names:
                entity = Entity(
                    name=name,
                    type=entity_type,
                    properties={"source": "PDF extraction"}
                )
                self.neo4j.add_entity(entity)
                entity_count += 1

        print(f"[OK] Added {entity_count} entities to graph")

        # Step 3: Extract relationships
        # Get chunks again for relationship extraction
        all_chunks = []
        for term in ["infant", "newborn", "treatment"]:
            results = self.opensearch.search(term, top_k=200)
            all_chunks.extend(results)

        unique_chunks = {c.chunk_id: c for c in all_chunks}.values()
        chunks = list(unique_chunks)[:limit_chunks] if limit_chunks else list(unique_chunks)

        relationships = self.extract_relationships_from_chunks(chunks, entities)

        # Step 4: Add relationships to Neo4j
        print(f"\n[INFO] Adding relationships to Neo4j...")
        rel_count = 0
        for source, target, rel_type in relationships:
            relationship = Relationship(
                source=source,
                target=target,
                rel_type=rel_type,
                properties={"source": "PDF extraction"}
            )
            if self.neo4j.add_relationship(relationship):
                rel_count += 1

        print(f"[OK] Added {rel_count} relationships to graph")

        # Step 5: Show stats
        print("\n" + "="*80)
        print("KNOWLEDGE GRAPH BUILT")
        print("="*80)
        stats = self.neo4j.get_stats()
        print(f"\nTotal nodes: {stats['total_nodes']}")
        print(f"Total relationships: {stats['total_relationships']}")
        print(f"\nNodes by type:")
        for label, count in stats['nodes'].items():
            if count > 0:
                print(f"  {label}: {count}")


if __name__ == "__main__":
    print("=== Medical Knowledge Graph Builder ===\n")

    # Initialize stores
    print("[Loading] OpenSearch...")
    opensearch = OpenSearchStore(
        host=settings.OPENSEARCH_HOST,
        port=settings.OPENSEARCH_PORT,
        index_name=settings.OPENSEARCH_INDEX
    )

    print("[Loading] Neo4j...")
    neo4j = Neo4jStore(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD
    )

    # Clear existing graph (optional)
    print("\n[WARN] Clearing existing graph...")
    neo4j.clear_graph()

    # Build KG
    builder = MedicalKGBuilder(opensearch, neo4j)
    builder.build_graph(limit_chunks=500)  # Process 500 chunks for speed

    # Cleanup
    opensearch.close()
    neo4j.close()

    print("\n[OK] Knowledge graph build complete!")
