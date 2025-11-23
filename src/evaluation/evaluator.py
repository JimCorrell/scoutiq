"""
Model evaluation and metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class RegressionEvaluator:
    """
    Evaluate regression models
    """

    def __init__(self):
        """Initialize regression evaluator"""
        if not SKLEARN_AVAILABLE:
            logger.warning(
                "scikit-learn not available. Limited evaluation capabilities."
            )
        logger.info("Initialized RegressionEvaluator")

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate regression predictions

        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: List of metrics to compute

        Returns:
            Dictionary of metric values
        """
        if not SKLEARN_AVAILABLE:
            return {}

        metrics = metrics or ["mae", "rmse", "r2", "mape"]
        results = {}

        if "mae" in metrics:
            results["mae"] = mean_absolute_error(y_true, y_pred)

        if "rmse" in metrics:
            results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))

        if "r2" in metrics:
            results["r2"] = r2_score(y_true, y_pred)

        if "mape" in metrics:
            results["mape"] = mean_absolute_percentage_error(y_true, y_pred)

        return results

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot predicted vs actual values

        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 6))

        plt.scatter(y_true, y_pred, alpha=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(title)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Plot",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot residuals

        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Optional path to save plot
        """
        residuals = y_true - y_pred

        plt.figure(figsize=(10, 6))

        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--", lw=2)

        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(title)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()

        plt.close()


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize model evaluator

        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.regression_evaluator = RegressionEvaluator()
        logger.info(f"Initialized ModelEvaluator with output directory: {output_dir}")

    def evaluate_models(
        self, models: Dict, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Evaluate all models on test set

        Args:
            models: Dictionary of models by target and model type
            X_test: Test features
            y_test: Test targets

        Returns:
            DataFrame with evaluation results
        """
        logger.info("Evaluating models on test set")

        results = []

        for target in y_test.columns:
            logger.info(f"Evaluating models for target: {target}")

            y_true = y_test[target].values
            target_models = models.get(target, {})

            for model_name, model in target_models.items():
                try:
                    y_pred = model.predict(X_test)

                    metrics = self.regression_evaluator.evaluate(y_true, y_pred)

                    result = {"target": target, "model": model_name, **metrics}
                    results.append(result)

                    logger.info(
                        f"{model_name} - MAE: {metrics.get('mae', 0):.4f}, "
                        f"RMSE: {metrics.get('rmse', 0):.4f}, "
                        f"R2: {metrics.get('r2', 0):.4f}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error evaluating {model_name} for {target}: {str(e)}"
                    )

        results_df = pd.DataFrame(results)

        # Save results
        results_file = self.output_dir / "model_evaluation.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"Saved evaluation results to {results_file}")

        return results_df

    def plot_feature_importance(
        self, model, model_name: str, target: str, top_n: int = 20
    ) -> None:
        """
        Plot feature importance

        Args:
            model: Trained model with feature importance
            model_name: Name of the model
            target: Target variable name
            top_n: Number of top features to plot
        """
        try:
            importance = model.get_feature_importance()

            plt.figure(figsize=(10, 8))
            importance.head(top_n).plot(kind="barh")
            plt.xlabel("Importance")
            plt.title(f"Top {top_n} Features - {model_name} ({target})")
            plt.tight_layout()

            save_path = (
                self.output_dir / f"feature_importance_{target}_{model_name}.png"
            )
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved feature importance plot to {save_path}")

            plt.close()

        except Exception as e:
            logger.warning(f"Could not plot feature importance: {str(e)}")

    def create_summary_report(self, results_df: pd.DataFrame) -> str:
        """
        Create summary report of model performance

        Args:
            results_df: DataFrame with evaluation results

        Returns:
            Summary report as string
        """
        logger.info("Creating summary report")

        report = []
        report.append("=" * 80)
        report.append("MODEL EVALUATION SUMMARY")
        report.append("=" * 80)
        report.append("")

        for target in results_df["target"].unique():
            target_results = results_df[results_df["target"] == target]

            report.append(f"\nTarget: {target}")
            report.append("-" * 80)

            for _, row in target_results.iterrows():
                report.append(f"\n{row['model'].upper()}")
                report.append(f"  MAE:  {row.get('mae', 0):.4f}")
                report.append(f"  RMSE: {row.get('rmse', 0):.4f}")
                report.append(f"  R²:   {row.get('r2', 0):.4f}")
                report.append(f"  MAPE: {row.get('mape', 0):.4f}")

            # Best model
            best_model = target_results.loc[target_results["mae"].idxmin()]
            report.append(
                f"\n  Best Model: {best_model['model']} (MAE: {best_model['mae']:.4f})"
            )

        report.append("\n" + "=" * 80)

        report_text = "\n".join(report)

        # Save report
        report_file = self.output_dir / "evaluation_summary.txt"
        with open(report_file, "w") as f:
            f.write(report_text)

        logger.info(f"Saved summary report to {report_file}")

        return report_text
