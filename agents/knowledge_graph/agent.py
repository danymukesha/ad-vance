"""AD-VANCE: Knowledge Graph Agent

This agent builds and maintains a dynamic Alzheimer's-specific knowledge graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import networkx as nx


logger = logging.getLogger(__name__)


ENTITY_TYPES = [
    "gene",
    "protein",
    "pathway",
    "drug",
    "disease",
    "phenotype",
    "biomarker",
    "brain_region",
    "cell_type",
    "clinical_trial",
]

RELATIONSHIP_TYPES = [
    "interacts_with",
    "regulates",
    "expressed_in",
    "associated_with",
    "treats",
    "causes",
    "has_function",
    "part_of",
    "localizes_to",
    "targets",
]


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    
    id: str
    name: str
    entity_type: str
    properties: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    
    source_id: str
    target_id: str
    relationship_type: str
    properties: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


class KnowledgeGraphAgent:
    """Agent for building and querying the Alzheimer's knowledge graph.
    
    Constructs a dynamic graph from genes, proteins, pathways, drugs, 
    phenotypes, and other relevant entities.
    """
    
    def __init__(self):
        """Initialize the Knowledge Graph Agent."""
        self.graph = nx.MultiDiGraph()
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        
        self._initialize_ad_genes()
        self._initialize_ad_pathways()
        self._initialize_drug_targets()
        
        logger.info("Knowledge Graph Agent initialized with AD entities")
    
    def _initialize_ad_genes(self) -> None:
        """Initialize known AD-associated genes."""
        ad_genes = {
            "APP": {"full_name": "Amyloid Precursor Protein", "chromosome": "21"},
            "PSEN1": {"full_name": "Presenilin 1", "chromosome": "14"},
            "PSEN2": {"full_name": "Presenilin 2", "chromosome": "1"},
            "APOE": {"full_name": "Apolipoprotein E", "chromosome": "19"},
            "BIN1": {"full_name": "Bridging Integrator 1", "chromosome": "2"},
            "CLU": {"full_name": "Clusterin", "chromosome": "8"},
            "CR1": {"full_name": "Complement Receptor 1", "chromosome": "1"},
            "PICALM": {"full_name": "Phosphatidylinositol Binding Clathrin Assembly Protein", "chromosome": "10"},
            "ABCA7": {"full_name": "ATP Binding Cassette Subfamily A Member 7", "chromosome": "19"},
            "MS4A4A": {"full_name": "Membrane Spanning 4-Domains A4A", "chromosome": "11"},
            "CD2AP": {"full_name": "CD2 Associated Protein", "chromosome": "6"},
            "EPHA1": {"full_name": "Ephrin Type-A Receptor 1", "chromosome": "7"},
            "SORL1": {"full_name": "Sortilin Related Receptor 1", "chromosome": "11"},
            "TREM2": {"full_name": "Triggering Receptor Expressed On Myeloid Cells 2", "chromosome": "6"},
            "TYROBP": {"full_name": "TYRO Protein Tyrosine Kinase Binding Protein", "chromosome": "19"},
            "CD33": {"full_name": "CD33 Molecule", "chromosome": "19"},
            "PLD3": {"full_name": "Phospholipase D3", "chromosome": "19"},
        }
        
        for gene_id, props in ad_genes.items():
            self.add_entity(
                id=gene_id,
                name=props["full_name"],
                entity_type="gene",
                properties=props,
                sources=["GWAS Catalog", "AMP-AD"],
            )
    
    def _initialize_ad_pathways(self) -> None:
        """Initialize AD-related pathways."""
        pathways = {
            "Amyloid_Processing": {
                "description": "Amyloid beta processing pathway",
                "genes": ["APP", "BACE1", "PSEN1", "PSEN2"],
            },
            "Tau_Phosphorylation": {
                "description": "Tau protein phosphorylation",
                "genes": ["MAPT", "GSK3B", "CDK5", "PPP2CA"],
            },
            "Inflammatory_Response": {
                "description": "Neuroinflammation pathway",
                "genes": ["TREM2", "TYROBP", "IL1B", "TNF"],
            },
            "Lipid_Metabolism": {
                "description": "Lipid metabolism in brain",
                "genes": ["APOE", "ABCA7", "ABCA1", "CLU"],
            },
            "Synaptic_Function": {
                "description": "Synaptic plasticity and function",
                "genes": ["BIN1", "PICALM", "SNAP25", "SYN1"],
            },
            "Autophagy": {
                "description": "Autophagy pathway",
                "genes": ["BECN1", "ATG5", "MTOR", "ULK1"],
            },
            "Mitochondrial_Function": {
                "description": "Mitochondrial dysfunction",
                "genes": ["TFAM", "PGC1A", "ND1", "COX1"],
            },
            "Oxidative_Stress": {
                "description": "Oxidative stress response",
                "genes": ["SOD1", "CAT", "GPX1", "NQO1"],
            },
        }
        
        for pathway_id, props in pathways.items():
            self.add_entity(
                id=pathway_id,
                name=props["description"],
                entity_type="pathway",
                properties=props,
                sources=["KEGG", "Reactome"],
            )
            
            for gene in props["genes"]:
                self.add_relationship(
                    source_id=gene,
                    target_id=pathway_id,
                    relationship_type="part_of",
                    sources=["KEGG"],
                )
    
    def _initialize_drug_targets(self) -> None:
        """Initialize AD drug targets."""
        drugs = {
            "Donepezil": {"target": "AChE", "status": "Approved"},
            "Rivastigmine": {"target": "AChE", "status": "Approved"},
            "Galantamine": {"target": "AChE", "status": "Approved"},
            "Memantine": {"target": "NMDA", "status": "Approved"},
            "Lecanemab": {"target": "Aβ", "status": "Approved"},
            "Donanemab": {"target": "Aβ", "status": "Approved"},
            "Aducanumab": {"target": "Aβ", "status": "Approved"},
            "Sodium_Olorate": {"target": "TREM2", "status": "Investigational"},
        }
        
        for drug_id, props in drugs.items():
            self.add_entity(
                id=drug_id,
                name=drug_id,
                entity_type="drug",
                properties=props,
                sources=["DrugBank", "ClinicalTrials.gov"],
            )
    
    def add_entity(
        self,
        id: str,
        name: str,
        entity_type: str,
        properties: dict | None = None,
        sources: list[str] | None = None,
        confidence: float = 1.0,
    ) -> Entity:
        """Add an entity to the knowledge graph."""
        entity = Entity(
            id=id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            sources=sources or [],
            confidence=confidence,
        )
        
        self.entities[id] = entity
        
        self.graph.add_node(
            id,
            name=name,
            entity_type=entity_type,
            properties=entity.properties,
        )
        
        logger.debug(f"Added entity: {id} ({entity_type})")
        return entity
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: dict | None = None,
        sources: list[str] | None = None,
        confidence: float = 1.0,
    ) -> Relationship:
        """Add a relationship to the knowledge graph."""
        if source_id not in self.entities:
            logger.warning(f"Source entity not found: {source_id}")
            return None
        
        if target_id not in self.entities:
            logger.warning(f"Target entity not found: {target_id}")
            return None
        
        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            properties=properties or {},
            sources=sources or [],
            confidence=confidence,
        )
        
        self.relationships.append(relationship)
        
        self.graph.add_edge(
            source_id,
            target_id,
            relationship_type=relationship_type,
            properties=relationship.properties,
        )
        
        logger.debug(f"Added relationship: {source_id} -> {target_id} ({relationship_type})")
        return relationship
    
    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return self.entities.get(entity_id)
    
    def get_neighbors(
        self,
        entity_id: str,
        relationship_type: str | None = None,
    ) -> list[tuple[Entity, str]]:
        """Get neighbors of an entity."""
        if entity_id not in self.graph:
            return []
        
        neighbors = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph.edges[entity_id, neighbor, 0]
            rel_type = edge_data.get("relationship_type", "unknown")
            
            if relationship_type is None or rel_type == relationship_type:
                entity = self.entities.get(neighbor)
                if entity:
                    neighbors.append((entity, rel_type))
        
        return neighbors
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 4,
    ) -> list[list[str]]:
        """Find paths between two entities."""
        if source_id not in self.graph or target_id not in self.graph:
            return []
        
        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=max_length,
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_subgraph(
        self,
        entity_ids: list[str],
        include_neighbors: bool = True,
    ) -> nx.Graph:
        """Get a subgraph for given entities."""
        if not include_neighbors:
            return self.graph.subgraph(entity_ids).copy()
        
        nodes = set(entity_ids)
        for entity_id in entity_ids:
            if entity_id in self.graph:
                nodes.update(self.graph.neighbors(entity_id))
        
        return self.graph.subgraph(nodes).copy()
    
    def execute(self, task_description: str, context: dict | None = None) -> dict:
        """Execute a knowledge graph task.
        
        Args:
            task_description: Description of the task
            context: Additional context
            
        Returns:
            Results from the knowledge graph operation
        """
        logger.info(f"Executing KG task: {task_description}")
        
        context = context or {}
        
        operation = context.get("operation", "stats")
        
        if operation == "stats":
            return self._get_statistics()
        elif operation == "search":
            return self._search_entities(context.get("query", ""))
        elif operation == "pathways":
            return self._get_pathway_genes(context.get("pathway_id", ""))
        elif operation == "enrich":
            return self._enrich_genes(context.get("genes", []))
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    def _get_statistics(self) -> dict:
        """Get knowledge graph statistics."""
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": {
                et: sum(1 for e in self.entities.values() if e.entity_type == et)
                for et in ENTITY_TYPES
            },
            "relationship_types": {
                rt: sum(1 for r in self.relationships if r.relationship_type == rt)
                for rt in RELATIONSHIP_TYPES
            },
            "graph_density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph.to_undirected()),
        }
    
    def _search_entities(self, query: str) -> dict:
        """Search for entities matching a query."""
        results = []
        
        query_lower = query.lower()
        
        for entity in self.entities.values():
            if query_lower in entity.name.lower() or query_lower in entity.id.lower():
                results.append({
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "confidence": entity.confidence,
                    "sources": entity.sources,
                })
        
        return {"query": query, "results": results, "count": len(results)}
    
    def _get_pathway_genes(self, pathway_id: str) -> dict:
        """Get genes in a pathway."""
        pathway = self.entities.get(pathway_id)
        
        if not pathway or pathway.entity_type != "pathway":
            return {"error": f"Pathway not found: {pathway_id}"}
        
        genes = []
        for neighbor, rel_type in self.get_neighbors(pathway_id, "part_of"):
            genes.append({
                "id": neighbor.id,
                "name": neighbor.name,
            })
        
        return {
            "pathway": pathway_id,
            "pathway_name": pathway.name,
            "genes": genes,
        }
    
    def _enrich_genes(self, genes: list[str]) -> dict:
        """Enrich input genes with pathway information."""
        enriched = []
        
        for gene_id in genes:
            gene = self.entities.get(gene_id)
            
            if gene:
                pathways = [
                    {
                        "id": p.id,
                        "name": p.name,
                    }
                    for p, _ in self.get_neighbors(gene_id, "part_of")
                ]
                
                enriched.append({
                    "gene": gene_id,
                    "pathways": pathways,
                })
        
        return {"enriched_genes": enriched}
    
    def to_networkx(self) -> nx.Graph:
        """Export knowledge graph as NetworkX graph."""
        return self.graph.copy()
    
    def to_cytoscape(self) -> list[dict]:
        """Export knowledge graph in Cytoscape.js format."""
        elements = []
        
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            elements.append({
                "data": {
                    "id": node,
                    "label": data.get("name", node),
                    "type": data.get("entity_type", "unknown"),
                }
            })
        
        for source, target, data in self.graph.edges(data=True):
            elements.append({
                "data": {
                    "source": source,
                    "target": target,
                    "relationship": data.get("relationship_type", "unknown"),
                }
            })
        
        return elements
