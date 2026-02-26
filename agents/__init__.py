"""AD-VANCE: Multi-Agent AI System for Alzheimer's Disease Research

This package contains the core agents for the AD-VANCE system.
"""

from .orchestrator import ResearchOrchestrator
from .data_integration import DataIntegrationAgent
from .knowledge_graph import KnowledgeGraphAgent
from .hypothesis_generation import HypothesisGenerationAgent
from .validation import ValidationAgent
from .reproducibility import ReproducibilityAgent

__version__ = "1.0.0"

__all__ = [
    "ResearchOrchestrator",
    "DataIntegrationAgent",
    "KnowledgeGraphAgent",
    "HypothesisGenerationAgent",
    "ValidationAgent",
    "ReproducibilityAgent",
]
