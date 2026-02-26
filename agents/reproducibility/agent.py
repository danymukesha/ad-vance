"""AD-VANCE: Reproducibility & Audit Agent

This agent tracks data lineage, logs decisions, and ensures transparency.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class DecisionLog:
    """Represents a logged decision."""
    
    id: str
    timestamp: datetime
    agent: str
    action: str
    inputs: dict
    outputs: dict
    reasoning: str
    metadata: dict = field(default_factory=dict)


@dataclass
class DataLineage:
    """Represents data lineage information."""
    
    dataset_id: str
    source: str
    transformations: list[str]
    version: str
    created_at: datetime
    hash: str | None = None


@dataclass
class PipelineRun:
    """Represents a pipeline execution."""
    
    run_id: str
    start_time: datetime
    end_time: datetime | None
    status: str
    tasks: list[dict]
    metrics: dict
    artifacts: list[str]
    errors: list[str] = field(default_factory=list)


class ReproducibilityAgent:
    """Agent for ensuring reproducibility and auditability.
    
    Tracks:
    - Data lineage
    - Decision logs
    - Pipeline runs
    - Model versions
    - Experiment tracking
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        enable_mlflow: bool = False,
    ):
        """Initialize the Reproducibility Agent.
        
        Args:
            log_dir: Directory for storing logs
            enable_mlflow: Whether to use MLflow for tracking
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_mlflow = enable_mlflow
        self.mlflow_client = None
        
        self.decision_logs: list[DecisionLog] = []
        self.data_lineage: dict[str, DataLineage] = {}
        self.pipeline_runs: list[PipelineRun] = []
        
        if enable_mlflow:
            self._setup_mlflow()
        
        logger.info("Reproducibility Agent initialized")
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow for experiment tracking."""
        try:
            import mlflow
            
            mlflow.set_experiment("ad-vance")
            self.mlflow_client = mlflow
            logger.info("MLflow tracking enabled")
        except ImportError:
            logger.warning("MLflow not available, using file-based logging")
            self.enable_mlflow = False
    
    def log_decision(
        self,
        agent: str,
        action: str,
        inputs: dict,
        outputs: dict,
        reasoning: str,
        metadata: dict | None = None,
    ) -> str:
        """Log a decision.
        
        Args:
            agent: Agent making the decision
            action: Action taken
            inputs: Input parameters
            outputs: Output results
            reasoning: Reasoning behind the decision
            metadata: Additional metadata
            
        Returns:
            Decision log ID
        """
        import uuid
        
        log_id = str(uuid.uuid4())[:8]
        
        decision = DecisionLog(
            id=log_id,
            timestamp=datetime.now(),
            agent=agent,
            action=action,
            inputs=inputs,
            outputs=outputs,
            reasoning=reasoning,
            metadata=metadata or {},
        )
        
        self.decision_logs.append(decision)
        
        self._save_decision_log(decision)
        
        if self.enable_mlflow and self.mlflow_client:
            self.mlflow_client.log_param(f"{agent}_action", action)
            self.mlflow_client.log_param(f"{agent}_inputs", json.dumps(inputs))
        
        logger.info(f"Logged decision {log_id}: {agent} - {action}")
        return log_id
    
    def _save_decision_log(self, decision: DecisionLog) -> None:
        """Save decision log to file."""
        log_file = self.log_dir / "decisions" / f"{decision.id}.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, "w") as f:
            json.dump(asdict(decision), f, indent=2, default=str)
    
    def track_data_lineage(
        self,
        dataset_id: str,
        source: str,
        transformations: list[str],
        version: str = "1.0",
    ) -> DataLineage:
        """Track data lineage.
        
        Args:
            dataset_id: Dataset identifier
            source: Original data source
            transformations: List of transformations applied
            version: Dataset version
            
        Returns:
            Data lineage object
        """
        import hashlib
        
        lineage = DataLineage(
            dataset_id=dataset_id,
            source=source,
            transformations=transformations,
            version=version,
            created_at=datetime.now(),
        )
        
        content = f"{dataset_id}{source}{transformations}{version}"
        lineage.hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        self.data_lineage[dataset_id] = lineage
        
        lineage_file = self.log_dir / "lineage" / f"{dataset_id}.json"
        lineage_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(lineage_file, "w") as f:
            json.dump(asdict(lineage), f, indent=2, default=str)
        
        logger.info(f"Tracked lineage for {dataset_id}")
        return lineage
    
    def start_pipeline_run(
        self,
        pipeline_name: str,
        config: dict,
    ) -> str:
        """Start tracking a pipeline run.
        
        Args:
            pipeline_name: Name of the pipeline
            config: Pipeline configuration
            
        Returns:
            Run ID
        """
        import uuid
        
        run_id = str(uuid.uuid4())[:8]
        
        run = PipelineRun(
            run_id=run_id,
            start_time=datetime.now(),
            end_time=None,
            status="started",
            tasks=[],
            metrics={},
            artifacts=[],
        )
        
        self.pipeline_runs.append(run)
        
        if self.enable_mlflow and self.mlflow_client:
            self.mlflow_client.start_run(run_id=f"ad-vance-{run_id}")
            self.mlflow_client.log_params(config)
        
        logger.info(f"Started pipeline run {run_id}: {pipeline_name}")
        return run_id
    
    def end_pipeline_run(
        self,
        run_id: str,
        status: str,
        metrics: dict | None = None,
        errors: list[str] | None = None,
    ) -> None:
        """End tracking a pipeline run.
        
        Args:
            run_id: Run ID
            status: Final status (success/failed)
            metrics: Run metrics
            errors: Errors encountered
        """
        for run in self.pipeline_runs:
            if run.run_id == run_id:
                run.end_time = datetime.now()
                run.status = status
                run.metrics = metrics or {}
                run.errors = errors or []
                
                if self.enable_mlflow and self.mlflow_client:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.mlflow_client.log_metric(
                                self.mlflow_client.Metric(key, value)
                            )
                    self.mlflow_client.end_run()
                
                self._save_pipeline_run(run)
                
                logger.info(f"Ended pipeline run {run_id}: {status}")
                break
    
    def _save_pipeline_run(self, run: PipelineRun) -> None:
        """Save pipeline run to file."""
        run_file = self.log_dir / "runs" / f"{run.run_id}.json"
        run_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(run_file, "w") as f:
            json.dump(asdict(run), f, indent=2, default=str)
    
    def get_decision_history(
        self,
        agent: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get decision history.
        
        Args:
            agent: Filter by agent
            limit: Maximum number of logs to return
            
        Returns:
            List of decision logs
        """
        logs = self.decision_logs
        
        if agent:
            logs = [l for l in logs if l.agent == agent]
        
        logs = sorted(logs, key=lambda x: x.timestamp, reverse=True)
        
        return [asdict(l) for l in logs[:limit]]
    
    def generate_transparency_report(self) -> dict:
        """Generate a transparency report.
        
        Returns:
            Transparency report
        """
        import platform
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "system_info": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
            "statistics": {
                "total_decisions": len(self.decision_logs),
                "total_datasets": len(self.data_lineage),
                "total_runs": len(self.pipeline_runs),
            },
            "agents": {},
            "data_sources": list(set(l.source for l in self.data_lineage.values())),
            "runs": {
                "completed": len([r for r in self.pipeline_runs if r.status == "success"]),
                "failed": len([r for r in self.pipeline_runs if r.status == "failed"]),
            },
        }
        
        for log in self.decision_logs:
            if log.agent not in report["agents"]:
                report["agents"][log.agent] = {"decisions": 0, "actions": set()}
            report["agents"][log.agent]["decisions"] += 1
            report["agents"][log.agent]["actions"].add(log.action)
        
        for agent in report["agents"]:
            report["agents"][agent]["actions"] = list(report["agents"][agent]["actions"])
        
        report_file = self.log_dir / "transparency_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Generated transparency report")
        return report
    
    def execute(self, task_description: str, context: dict | None = None) -> dict:
        """Execute a reproducibility task.
        
        Args:
            task_description: Description of the task
            context: Additional context
            
        Returns:
            Results
        """
        logger.info(f"Executing reproducibility task: {task_description}")
        
        context = context or {}
        
        operation = context.get("operation", "stats")
        
        if operation == "log_decision":
            return {
                "log_id": self.log_decision(
                    agent=context.get("agent", "system"),
                    action=context.get("action", "unknown"),
                    inputs=context.get("inputs", {}),
                    outputs=context.get("outputs", {}),
                    reasoning=context.get("reasoning", ""),
                )
            }
        
        elif operation == "track_lineage":
            return {
                "lineage": asdict(self.track_data_lineage(
                    dataset_id=context.get("dataset_id"),
                    source=context.get("source"),
                    transformations=context.get("transformations", []),
                    version=context.get("version", "1.0"),
                ))
            }
        
        elif operation == "start_run":
            return {
                "run_id": self.start_pipeline_run(
                    pipeline_name=context.get("pipeline_name", "default"),
                    config=context.get("config", {}),
                )
            }
        
        elif operation == "end_run":
            self.end_pipeline_run(
                run_id=context.get("run_id"),
                status=context.get("status", "success"),
                metrics=context.get("metrics", {}),
                errors=context.get("errors", []),
            )
            return {"status": "success"}
        
        elif operation == "report":
            return self.generate_transparency_report()
        
        elif operation == "history":
            return {
                "decisions": self.get_decision_history(
                    agent=context.get("agent"),
                    limit=context.get("limit", 100),
                )
            }
        
        else:
            return {
                "total_decisions": len(self.decision_logs),
                "total_datasets": len(self.data_lineage),
                "total_runs": len(self.pipeline_runs),
            }
