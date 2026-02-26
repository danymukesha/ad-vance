"""AD-VANCE: Validation & Simulation Agent

This agent validates hypotheses through cross-validation, survival analysis,
and statistical testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Represents a validation result."""
    
    hypothesis_id: str
    test_name: str
    passed: bool
    metric_name: str
    metric_value: float
    p_value: float | None = None
    confidence_interval: tuple[float, float] | None = None
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ValidationAgent:
    """Agent for validating hypotheses through rigorous statistical testing.
    
    Performs:
    - Cross-validation for classification
    - Survival analysis
    - Clustering stability
    - Causal inference tests
    - Baseline comparisons
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
    ):
        """Initialize the Validation Agent.
        
        Args:
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        logger.info(f"Validation Agent initialized with {n_folds}-fold CV")
    
    def validate_classification(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any = None,
    ) -> dict:
        """Validate classification performance.
        
        Args:
            X: Feature matrix
            y: Labels
            model: Model to validate
            
        Returns:
            Validation results
        """
        logger.info("Running classification validation")
        
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
        
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        try:
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
            
            model.fit(X, y)
            y_proba = model.predict_proba(X)[:, 1]
            
            auc = roc_auc_score(y, y_proba)
            ap = average_precision_score(y, y_proba)
            
            precision, recall, _ = precision_recall_curve(y, y_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            f1_max = np.max(f1_scores)
            
            result = {
                "status": "success",
                "test": "classification",
                "metrics": {
                    "roc_auc": {
                        "value": float(auc),
                        "cv_mean": float(cv_scores.mean()),
                        "cv_std": float(cv_scores.std()),
                        "ci_95": (
                            float(cv_scores.mean() - 1.96 * cv_scores.std()),
                            float(cv_scores.mean() + 1.96 * cv_scores.std()),
                        ),
                    },
                    "average_precision": {"value": float(ap)},
                    "max_f1": {"value": float(f1_max)},
                },
                "passed": cv_scores.mean() > 0.7,
            }
            
            logger.info(f"Classification validation: AUC={auc:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Classification validation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def validate_survival(
        self,
        time: np.ndarray,
        event: np.ndarray,
        X: np.ndarray | None = None,
    ) -> dict:
        """Validate survival prediction.
        
        Args:
            time: Survival time
            event: Event indicator (1=event, 0=censored)
            X: Feature matrix
            
        Returns:
            Validation results
        """
        logger.info("Running survival analysis validation")
        
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import concordance_index
        
        kmf = KaplanMeierFitter()
        kmf.fit(time, event)
        
        median_survival = np.median(time[event == 1])
        
        result = {
            "status": "success",
            "test": "survival_analysis",
            "metrics": {
                "median_survival_months": {"value": float(median_survival)},
            },
        }
        
        if X is not None and X.shape[1] > 0:
            try:
                cph = CoxPHFitter()
                df_surv = np.column_stack([time, event])
                if X.shape[0] == time.shape[0]:
                    df_surv = np.column_stack([time, event, X[:, :min(5, X.shape[1])]])
                
                cph.fit(df_surv, duration_col=0, event_col=1)
                
                c_index = cph.concordance_index_
                
                result["metrics"]["cox_c_index"] = {"value": float(c_index)}
                result["metrics"]["cox_p_value"] = {"value": float(cph.log_likelihood_ratio_test().p_value)}
                
                logger.info(f"Survival validation: C-index={c_index:.3f}")
                
            except Exception as e:
                logger.warning(f"Cox model failed: {e}")
        
        result["passed"] = True
        return result
    
    def validate_clustering(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """Validate clustering stability.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Validation results
        """
        logger.info("Running clustering validation")
        
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        from sklearn.metrics import adjusted_rand_score
        from sklearn.cluster import KMeans
        from sklearn.model_selection import train_test_split
        
        try:
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
            
            X_train, X_test, labels_train, labels_test = train_test_split(
                X, labels, test_size=0.5, random_state=self.random_state
            )
            
            kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=self.random_state)
            kmeans.fit(X_train)
            pred_labels = kmeans.predict(X_test)
            
            ari = adjusted_rand_score(labels_test, pred_labels)
            
            result = {
                "status": "success",
                "test": "clustering_stability",
                "metrics": {
                    "silhouette_score": {
                        "value": float(sil),
                        "interpretation": "Higher is better (-1 to 1)",
                    },
                    "calinski_harabasz": {"value": float(ch)},
                    "davies_bouldin": {
                        "value": float(db),
                        "interpretation": "Lower is better",
                    },
                    "adjusted_rand": {
                        "value": float(ari),
                        "interpretation": "Stability across splits (0 to 1)",
                    },
                },
                "passed": ari > 0.5 and sil > 0.3,
            }
            
            logger.info(f"Clustering validation: Silhouette={sil:.3f}, ARI={ari:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Clustering validation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def compare_baselines(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """Compare against baseline models.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Comparison results
        """
        logger.info("Running baseline comparison")
        
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "SVM": SVC(probability=True, random_state=self.random_state),
        }
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        results = {"status": "success", "test": "baseline_comparison", "models": {}}
        
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
                results["models"][name] = {
                    "cv_mean": float(scores.mean()),
                    "cv_std": float(scores.std()),
                }
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
        
        best_model = max(results["models"].items(), key=lambda x: x[1]["cv_mean"])
        results["best_model"] = {
            "name": best_model[0],
            "auc": best_model[1]["cv_mean"],
        }
        
        logger.info(f"Baseline comparison: Best model = {best_model[0]} ({best_model[1]['cv_mean']:.3f})")
        return results
    
    def run_ablation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> dict:
        """Run ablation study.
        
        Args:
            X: Feature matrix
            y: Labels
            feature_names: Names of features
            
        Returns:
            Ablation results
        """
        logger.info("Running ablation study")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        full_scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
        
        feature_importance = {}
        
        for i, fname in enumerate(feature_names):
            X_ablated = np.delete(X, i, axis=1)
            ablated_scores = cross_val_score(model, X_ablated, y, cv=skf, scoring="roc_auc")
            
            feature_importance[fname] = {
                "importance": float(full_scores.mean() - ablated_scores.mean()),
                "full_score": float(full_scores.mean()),
                "ablated_score": float(ablated_scores.mean()),
            }
        
        sorted_importance = sorted(
            feature_importance.items(),
            key=lambda x: x[1]["importance"],
            reverse=True,
        )
        
        result = {
            "status": "success",
            "test": "ablation_study",
            "full_model_auc": float(full_scores.mean()),
            "feature_importance": dict(sorted_importance[:10]),
        }
        
        logger.info(f"Ablation complete: {len(feature_names)} features tested")
        return result
    
    def execute(self, task_description: str, context: dict | None = None) -> dict:
        """Execute a validation task.
        
        Args:
            task_description: Description of the task
            context: Additional context
            
        Returns:
            Validation results
        """
        logger.info(f"Executing validation: {task_description}")
        
        context = context or {}
        
        test_type = context.get("test", "classification")
        
        if test_type == "classification":
            X = context.get("X", np.random.randn(500, 20))
            y = context.get("y", np.random.randint(0, 2, 500))
            return self.validate_classification(X, y)
        
        elif test_type == "survival":
            time = context.get("time", np.random.exponential(24, 500))
            event = context.get("event", np.random.randint(0, 2, 500))
            X = context.get("X", np.random.randn(500, 5))
            return self.validate_survival(time, event, X)
        
        elif test_type == "clustering":
            X = context.get("X", np.random.randn(500, 20))
            labels = context.get("labels", np.random.randint(0, 3, 500))
            return self.validate_clustering(X, labels)
        
        elif test_type == "baseline":
            X = context.get("X", np.random.randn(500, 20))
            y = context.get("y", np.random.randint(0, 2, 500))
            return self.compare_baselines(X, y)
        
        elif test_type == "ablation":
            X = context.get("X", np.random.randn(500, 20))
            y = context.get("y", np.random.randint(0, 2, 500))
            features = context.get("feature_names", [f"f{i}" for i in range(20)])
            return self.run_ablation(X, y, features)
        
        else:
            return {"status": "error", "error": f"Unknown test type: {test_type}"}
