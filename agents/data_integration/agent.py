"""AD-VANCE: Data Integration Agent

This agent handles data retrieval, cleaning, and harmonization
from various Alzheimer's disease data sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Represents a data source."""
    
    name: str
    source_type: str
    url: str
    description: str
    data_schema: dict | None = None
    last_updated: datetime | None = None
    record_count: int = 0


@dataclass
class Dataset:
    """Represents a collected dataset."""
    
    id: str
    name: str
    source: str
    data: pd.DataFrame | None = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    schema_version: str = "1.0"
    validation_status: str = "pending"


class DataIntegrationAgent:
    """Agent for integrating Alzheimer's disease data from multiple sources.
    
    Handles data retrieval, cleaning, normalization, and harmonization
    across heterogeneous datasets.
    """
    
    def __init__(
        self,
        cache_dir: str = "./data/cache",
        use_cache: bool = True,
    ):
        """Initialize the Data Integration Agent.
        
        Args:
            cache_dir: Directory for caching downloaded data
            use_cache: Whether to use cached data
        """
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        self.connectors: dict[str, Any] = {}
        self.datasets: dict[str, Dataset] = {}
        
        self._register_connectors()
        
        logger.info("Data Integration Agent initialized")
    
    def _register_connectors(self) -> None:
        """Register data source connectors."""
        from .connectors.adni import ADNIConnector
        from .connectors.ampad import AMPADConnector
        from .connectors.geo import GEOConnector
        from .connectors.rosmap import ROSMAPConnector
        
        self.connectors["adni"] = ADNIConnector()
        self.connectors["ampad"] = AMPADConnector()
        self.connectors["geo"] = GEOConnector()
        self.connectors["rosmap"] = ROSMAPConnector()
        
        logger.info(f"Registered {len(self.connectors)} data connectors")
    
    def _validate_schema(self, data: pd.DataFrame, expected_schema: dict) -> bool:
        """Validate data against expected schema."""
        for col, dtype in expected_schema.items():
            if col not in data.columns:
                logger.warning(f"Missing column: {col}")
                return False
            if not pd.api.types.is_dtype_equal(data[col].dtype, dtype):
                logger.warning(f"Type mismatch for {col}: expected {dtype}, got {data[col].dtype}")
        return True
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data."""
        df = data.copy()
        
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.strip()
                df[col] = df[col].replace("", pd.NA)
                df[col] = df[col].replace("NA", pd.NA)
                df[col] = df[col].replace("null", pd.NA)
        
        df = df.dropna(axis=1, how="all")
        
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _harmonize_datasets(
        self,
        datasets: list[Dataset],
        harmonization_schema: dict,
    ) -> pd.DataFrame:
        """Harmonize multiple datasets to a common schema."""
        harmonized_dfs = []
        
        for dataset in datasets:
            if dataset.data is None:
                continue
            
            df = dataset.data.copy()
            
            for target_col, source_cols in harmonization_schema.items():
                if isinstance(source_cols, str):
                    source_cols = [source_cols]
                
                for source_col in source_cols:
                    if source_col in df.columns:
                        df[target_col] = df[source_col]
                        break
            
            common_cols = list(harmonization_schema.keys())
            available_cols = [c for c in common_cols if c in df.columns]
            
            if available_cols:
                harmonized_dfs.append(df[available_cols])
        
        if harmonized_dfs:
            result = pd.concat(harmonized_dfs, ignore_index=True)
            return result
        
        return pd.DataFrame()
    
    def execute(self, task_description: str, context: dict | None = None) -> dict:
        """Execute a data integration task.
        
        Args:
            task_description: Description of the data task
            context: Additional context
            
        Returns:
            Dictionary with collected and processed data
        """
        logger.info(f"Executing data integration: {task_description}")
        
        context = context or {}
        
        target_sources = context.get("sources", ["adni", "ampad", "rosmap", "geo"])
        
        collected_datasets = []
        
        for source_name in target_sources:
            connector = self.connectors.get(source_name)
            if connector:
                try:
                    dataset = connector.fetch()
                    if dataset:
                        collected_datasets.append(dataset)
                        logger.info(f"Collected data from {source_name}: {len(dataset)} records")
                except Exception as e:
                    logger.error(f"Failed to fetch from {source_name}: {e}")
        
        if not collected_datasets:
            return {
                "status": "no_data",
                "message": "No data could be collected from specified sources",
                "datasets": [],
            }
        
        harmonized_data = self._harmonize_datasets(
            collected_datasets,
            context.get("harmonization_schema", {}),
        )
        
        cleaned_data = self._clean_data(harmonized_data)
        
        result = {
            "status": "success",
            "datasets": [ds.to_dict() if hasattr(ds, 'to_dict') else str(ds) for ds in collected_datasets],
            "harmonized_records": len(cleaned_data),
            "columns": list(cleaned_data.columns),
            "data_summary": cleaned_data.describe().to_dict() if not cleaned_data.empty else {},
        }
        
        logger.info(f"Data integration complete: {len(cleaned_data)} harmonized records")
        return result
    
    def collect_adni_data(self) -> Dataset:
        """Collect ADNI data."""
        connector = self.connectors.get("adni")
        if connector:
            return connector.fetch()
        raise ValueError("ADNI connector not available")
    
    def collect_ampad_data(self) -> Dataset:
        """Collect AMP-AD data."""
        connector = self.connectors.get("ampad")
        if connector:
            return connector.fetch()
        raise ValueError("AMP-AD connector not available")
    
    def collect_rosmap_data(self) -> Dataset:
        """Collect ROSMAP data."""
        connector = self.connectors.get("rosmap")
        if connector:
            return connector.fetch()
        raise ValueError("ROSMAP connector not available")
    
    def search_geo(self, query: str, max_results: int = 10) -> list[dict]:
        """Search GEO for relevant datasets."""
        connector = self.connectors.get("geo")
        if connector:
            return connector.search(query, max_results)
        return []
    
    def get_dataset(self, dataset_id: str) -> Dataset | None:
        """Get a dataset by ID."""
        return self.datasets.get(dataset_id)
    
    def list_datasets(self) -> list[Dataset]:
        """List all collected datasets."""
        return list(self.datasets.values())
