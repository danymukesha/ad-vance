"""ADNI Data Connector

Connector for Alzheimer's Disease Neuroimaging Initiative (ADNI) data.
ADNI is a major public dataset for Alzheimer's research.
"""

import logging
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


class ADNIConnector:
    """Connector for ADNI (Alzheimer's Disease Neuroimaging Initiative).
    
    ADNI is a longitudinal study that collects:
    - MRI and PET imaging data
    - Clinical assessments
    - Genetic data (genotyping)
    - Biomarkers (CSF, blood)
    """
    
    BASE_URL = "https://adni.loni.usc.edu"
    
    def __init__(self, cache_dir: str = "./data/cache"):
        """Initialize ADNI connector.
        
        Args:
            cache_dir: Directory for caching data
        """
        self.cache_dir = cache_dir
        self._data: pd.DataFrame | None = None
    
    def fetch(self, data_type: str = "clinical") -> pd.DataFrame:
        """Fetch ADNI data.
        
        Args:
            data_type: Type of data to fetch (clinical, imaging, genetic, biomarker)
            
        Returns:
            DataFrame with ADNI data
        """
        logger.info(f"Fetching ADNI {data_type} data")
        
        if data_type == "clinical":
            return self._get_clinical_data()
        elif data_type == "imaging":
            return self._get_imaging_data()
        elif data_type == "genetic":
            return self._get_genetic_data()
        elif data_type == "biomarker":
            return self._get_biomarker_data()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _get_clinical_data(self) -> pd.DataFrame:
        """Generate mock clinical data for demonstration.
        
        In production, this would connect to ADNI's actual API.
        """
        import numpy as np
        
        np.random.seed(42)
        
        n_samples = 1000
        
        data = {
            "RID": range(1, n_samples + 1),
            "AGE": np.random.randint(55, 90, n_samples),
            "PTGENDER": np.random.choice(["Male", "Female"], n_samples),
            "PTEDUCAT": np.random.randint(8, 20, n_samples),
            "APOE4": np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
            "MMSE": np.random.normal(27, 3, n_samples).clip(15, 30),
            "CDRSB": np.random.exponential(2, n_samples),
            "ADAS11": np.random.normal(15, 8, n_samples).clip(0, 70),
            "ADAS13": np.random.normal(20, 10, n_samples).clip(0, 85),
            "FAQ": np.random.exponential(5, n_samples),
            "DX": np.random.choice(
                ["CN", "MCI", "AD"],
                n_samples,
                p=[0.3, 0.4, 0.3]
            ),
            "VISCODE": np.random.choice(
                ["bl", "m06", "m12", "m24", "m36"],
                n_samples,
                p=[0.3, 0.2, 0.2, 0.2, 0.1]
            ),
        }
        
        df = pd.DataFrame(data)
        
        logger.info(f"Generated {len(df)} clinical records")
        return df
    
    def _get_imaging_data(self) -> pd.DataFrame:
        """Generate mock imaging data for demonstration."""
        import numpy as np
        
        np.random.seed(43)
        
        n_samples = 500
        
        data = {
            "RID": range(1, n_samples + 1),
            "VISCODE": np.random.choice(["bl", "m06", "m12", "m24"], n_samples),
            "HIPPO": np.random.normal(6500, 1000, n_samples),
            "ENTORHINAL": np.random.normal(3500, 500, n_samples),
            "MIDDLE_TEMPORAL": np.random.normal(18000, 2000, n_samples),
            "AMYGDALA": np.random.normal(2500, 400, n_samples),
            "WHITE_MATTER": np.random.normal(450000, 50000, n_samples),
            "CORTICAL_THICKNESS": np.random.normal(2.5, 0.3, n_samples),
            "FDG_PET": np.random.normal(7.5, 1.0, n_samples),
            "AV45_PET": np.random.normal(1.2, 0.3, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        logger.info(f"Generated {len(df)} imaging records")
        return df
    
    def _get_genetic_data(self) -> pd.DataFrame:
        """Generate mock genetic data for demonstration."""
        import numpy as np
        
        np.random.seed(44)
        
        n_samples = 1000
        
        data = {
            "RID": range(1, n_samples + 1),
            "APOE4": np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
            "BIN1": np.random.choice(["CC", "CT", "TT"], n_samples),
            "CLU": np.random.choice(["AA", "AG", "GG"], n_samples),
            "CR1": np.random.choice(["AA", "AG", "GG"], n_samples),
            "PICALM": np.random.choice(["CC", "CT", "TT"], n_samples),
            "ABCA7": np.random.choice(["AA", "AG", "GG"], n_samples),
            "MS4A4A": np.random.choice(["CC", "CT", "TT"], n_samples),
            "CD2AP": np.random.choice(["AA", "AG", "GG"], n_samples),
            "EPHA1": np.random.choice(["AA", "AG", "GG"], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        logger.info(f"Generated {len(df)} genetic records")
        return df
    
    def _get_biomarker_data(self) -> pd.DataFrame:
        """Generate mock biomarker data for demonstration."""
        import numpy as np
        
        np.random.seed(45)
        
        n_samples = 800
        
        data = {
            "RID": range(1, n_samples + 1),
            "VISCODE": np.random.choice(["bl", "m06", "m12", "m24"], n_samples),
            "ABETA": np.random.normal(200, 80, n_samples),
            "TAU": np.random.normal(300, 100, n_samples),
            "PTAU": np.random.normal(60, 20, n_samples),
            "NFL": np.random.normal(50, 25, n_samples),
            "GFAP": np.random.normal(100, 40, n_samples),
            "YKL40": np.random.normal(150, 50, n_samples),
            "IL6": np.random.normal(3, 1.5, n_samples),
            "TNF_ALPHA": np.random.normal(8, 3, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        logger.info(f"Generated {len(df)} biomarker records")
        return df
    
    def get_metadata(self) -> dict[str, Any]:
        """Get ADNI metadata."""
        return {
            "name": "Alzheimer's Disease Neuroimaging Initiative",
            "abbreviation": "ADNI",
            "description": "Longitudinal study for Alzheimer's disease biomarkers",
            "url": self.BASE_URL,
            "data_types": ["clinical", "imaging", "genetic", "biomarker"],
            "participants": 2000,
        }
