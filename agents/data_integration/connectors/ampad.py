"""AMP-AD Data Connector

Connector for Accelerating Medicines Partnership: Alzheimer's Disease (AMP-AD)
knowledge portal data.
"""

import logging
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


class AMPADConnector:
    """Connector for AMP-AD (Accelerating Medicines Partnership: Alzheimer's Disease).
    
    AMP-AD provides:
    - Multi-omic data (genomics, transcriptomics, proteomics)
    - Network models
    - Drug target information
    """
    
    BASE_URL = "https://ampadportal.org"
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """Initialize AMP-AD connector.
        
        Args:
            cache_dir: Directory for caching data
        """
        self.cache_dir = cache_dir
    
    def fetch(self, data_type: str = "proteomics") -> pd.DataFrame:
        """Fetch AMP-AD data.
        
        Args:
            data_type: Type of data (proteomics, transcriptomics, genomics, network)
            
        Returns:
            DataFrame with AMP-AD data
        """
        logger.info(f"Fetching AMP-AD {data_type} data")
        
        if data_type == "proteomics":
            return self._get_proteomics_data()
        elif data_type == "transcriptomics":
            return self._get_transcriptomics_data()
        elif data_type == "genomics":
            return self._get_genomics_data()
        elif data_type == "network":
            return self._get_network_data()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _get_proteomics_data(self) -> pd.DataFrame:
        """Generate mock proteomics data."""
        import numpy as np
        
        np.random.seed(52)
        
        n_samples = 600
        n_proteins = 200
        
        protein_ids = [f"PROT_{i:04d}" for i in range(n_proteins)]
        
        data = {
            "sample_id": [f"AMPAD_{i:04d}" for i in range(n_samples)],
            "brain_region": np.random.choice(
                ["DLPFC", "Temporal", "Parietal", "Cerebellum"],
                n_samples
            ),
            "diagnosis": np.random.choice(
                ["Control", "MCI", "AD"],
                n_samples,
                p=[0.35, 0.30, 0.35]
            ),
            "age_death": np.random.randint(60, 100, n_samples),
            "sex": np.random.choice(["M", "F"], n_samples),
        }
        
        for protein in protein_ids[:50]:
            data[protein] = np.random.normal(0, 1, n_samples)
        
        df = pd.DataFrame(data)
        
        logger.info(f"Generated proteomics data: {len(df)} samples, {len(protein_ids[:50])} proteins")
        return df
    
    def _get_transcriptomics_data(self) -> pd.DataFrame:
        """Generate mock transcriptomics data."""
        import numpy as np
        
        np.random.seed(53)
        
        n_samples = 800
        n_genes = 500
        
        gene_ids = [f"ENSG_{i:010d}" for i in range(n_genes)]
        
        data = {
            "sample_id": [f"AMPAD_T{ i:04d}" for i in range(n_samples)],
            "brain_region": np.random.choice(
                ["DLPFC", "Temporal", "Parietal", "Cerebellum", "Hippocampus"],
                n_samples
            ),
            "diagnosis": np.random.choice(
                ["Control", "MCI", "AD"],
                n_samples,
                p=[0.3, 0.3, 0.4]
            ),
            "age_death": np.random.randint(60, 100, n_samples),
            "sex": np.random.choice(["M", "F"], n_samples),
            "PMI": np.random.uniform(2, 12, n_samples),
        }
        
        for gene in gene_ids[:100]:
            data[gene] = np.random.normal(5, 2, n_samples)
        
        df = pd.DataFrame(data)
        
        logger.info(f"Generated transcriptomics data: {len(df)} samples")
        return df
    
    def _get_genomics_data(self) -> pd.DataFrame:
        """Generate mock genomics data (eQTLs)."""
        import numpy as np
        
        np.random.seed(54)
        
        n_samples = 1000
        
        data = {
            "sample_id": [f"AMPAD_G{i:04d}" for i in range(n_samples)],
            "diagnosis": np.random.choice(
                ["Control", "MCI", "AD"],
                n_samples,
                p=[0.3, 0.3, 0.4]
            ),
            "rs429358": np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]),
            "rs7412": np.random.choice([0, 1, 2], n_samples, p=[0.8, 0.15, 0.05]),
            "rs3851179": np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]),
            "rs744373": np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.35, 0.15]),
            "rs9331888": np.random.choice([0, 1, 2], n_samples, p=[0.45, 0.4, 0.15]),
            "rs3764650": np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.25, 0.05]),
            "rs6102056": np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.35, 0.15]),
            "rs10948363": np.random.choice([0, 1, 2], n_samples, p=[0.55, 0.35, 0.1]),
        }
        
        df = pd.DataFrame(data)
        
        logger.info(f"Generated genomics data: {len(df)} samples")
        return df
    
    def _get_network_data(self) -> pd.DataFrame:
        """Generate mock protein-protein interaction network data."""
        import numpy as np
        
        np.random.seed(55)
        
        n_interactions = 500
        
        ad_genes = [
            "APP", "PSEN1", "PSEN2", "APOE", "BIN1", "CLU", "CR1", "PICALM",
            "ABCA7", "MS4A4A", "CD2AP", "EPHA1", "PTK2B", "SORL1", "FERMT2",
            "SLC24A4", "RIN3", "DSG2", "INPP5D", "MEF2C", "NME8", "ZYX",
            "HLA-DRB1", "TREM2", "TYROBP", "PLD3", "TWNK", "OGG1", "MTHFR"
        ]
        
        interactions = []
        for _ in range(n_interactions):
            gene1 = np.random.choice(ad_genes)
            gene2 = np.random.choice(ad_genes)
            if gene1 != gene2:
                interactions.append((gene1, gene2, np.random.uniform(0.5, 1.0)))
        
        df = pd.DataFrame(
            interactions[:n_interactions],
            columns=["source", "target", "weight"]
        ).drop_duplicates()
        
        logger.info(f"Generated network data: {len(df)} interactions")
        return df
    
    def get_metadata(self) -> dict[str, Any]:
        """Get AMP-AD metadata."""
        return {
            "name": "Accelerating Medicines Partnership: Alzheimer's Disease",
            "abbreviation": "AMP-AD",
            "description": "Multi-omic data for Alzheimer's disease research",
            "url": self.BASE_URL,
            "data_types": ["proteomics", "transcriptomics", "genomics", "network"],
            "cohorts": ["Banner", "MSBB", "ROSMAP", "Mayo"],
        }
