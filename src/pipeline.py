"""
Main pipeline for baseball prospect projections
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
import yaml

from .data_ingestion import StructuredDataLoader, UnstructuredDataLoader, DataIntegrator
from .nlp import NLPPipeline
from .features import FeaturePipeline
from .models import ModelTrainer
from .evaluation import ModelEvaluator
from .utils import load_config, setup_logger

logger = setup_logger(__name__)


class ProspectProjectionPipeline:
    """
    End-to-end pipeline for prospect projections
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize projection pipeline

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.structured_loader = StructuredDataLoader(self.config["data"]["raw_dir"])
        self.unstructured_loader = UnstructuredDataLoader(
            self.config["data"]["raw_dir"]
        )
        self.data_integrator = DataIntegrator()
        self.nlp_pipeline = NLPPipeline()
        self.feature_pipeline = FeaturePipeline()
        self.model_trainer = ModelTrainer(self.config["models"])
        self.evaluator = ModelEvaluator()

        self.data = None
        self.features = None
        self.models = None
        self.ensembles = None

        logger.info("Initialized ProspectProjectionPipeline")

    def load_data(
        self, stats_file: Optional[str] = None, reports_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load structured and unstructured data

        Args:
            stats_file: CSV file with prospect statistics
            reports_file: CSV/JSON file with scouting reports

        Returns:
            Merged DataFrame
        """
        logger.info("Loading data")

        # Load structured data
        stats_file = stats_file or self.config["data"]["structured_data"]
        structured_df = self.structured_loader.load_csv(stats_file)
        structured_df = self.structured_loader.clean_structured_data(structured_df)

        # Load unstructured data
        reports_file = reports_file or self.config["data"]["unstructured_data"]
        unstructured_df = self.unstructured_loader.load_scouting_reports(reports_file)
        unstructured_df = self.unstructured_loader.validate_text_data(unstructured_df)

        # Merge data
        self.data = self.data_integrator.merge_data(structured_df, unstructured_df)

        logger.info(f"Loaded data with shape: {self.data.shape}")
        return self.data

    def process_nlp(self) -> pd.DataFrame:
        """
        Process scouting reports with NLP

        Returns:
            DataFrame with NLP features
        """
        logger.info("Processing scouting reports with NLP")

        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Process reports
        if "combined_reports" in self.data.columns:
            nlp_features = self.nlp_pipeline.process_dataframe(
                self.data, text_col="combined_reports"
            )

            # Merge NLP features with data
            self.data = self.data.merge(nlp_features, on="player_id", how="left")

        logger.info("NLP processing complete")
        return self.data

    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer features from structured and NLP data

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features")

        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.features = self.feature_pipeline.engineer_features(
            self.data,
            create_interactions=self.config["features"]["create_interactions"],
            scale_features=False,
        )

        logger.info(
            f"Feature engineering complete. Total features: {len(self.features.columns)}"
        )
        return self.features

    def prepare_training_data(self) -> tuple:
        """
        Prepare training and test sets

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing training data")

        if self.features is None:
            raise ValueError("Features not engineered. Call engineer_features() first.")

        # Select target columns
        target_cols = self.config["projections"].get(
            "batting_targets", []
        ) + self.config["projections"].get("pitching_targets", [])

        # Filter to only targets that exist in the data
        target_cols = [col for col in target_cols if col in self.features.columns]

        if not target_cols:
            raise ValueError("No target columns found in data")

        # Feature columns (exclude targets, IDs, text)
        exclude_cols = target_cols + ["player_id", "cleaned_text", "combined_reports"]
        feature_cols = [
            col
            for col in self.features.columns
            if col not in exclude_cols
            and self.features[col].dtype in ["int64", "float64"]
        ]

        # Remove rows with missing targets
        valid_data = self.features.dropna(subset=target_cols)

        # Split data
        from sklearn.model_selection import train_test_split

        X = valid_data[feature_cols]
        y = valid_data[target_cols]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["data"]["test_size"],
            random_state=self.config["data"]["random_seed"],
        )

        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict:
        """
        Train all models

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Dictionary of trained models
        """
        logger.info("Training models")

        # Split training data for validation
        from sklearn.model_selection import train_test_split

        x_tr, x_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.config["data"]["validation_size"],
            random_state=self.config["data"]["random_seed"],
        )

        # Train models
        self.models = self.model_trainer.train_models(x_tr, y_tr, x_val, y_val)

        # Create ensembles
        self.ensembles = self.model_trainer.create_ensembles()

        logger.info("Model training complete")
        return self.models

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate models on test set

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            DataFrame with evaluation results
        """
        logger.info("Evaluating models")

        if self.models is None:
            raise ValueError("Models not trained. Call train() first.")

        results = self.evaluator.evaluate_models(self.models, X_test, y_test)

        # Generate plots for best models
        for target in y_test.columns:
            target_results = results[results["target"] == target]
            best_model_name = target_results.loc[
                target_results["mae"].idxmin(), "model"
            ]
            best_model = self.models[target][best_model_name]

            self.evaluator.plot_feature_importance(best_model, best_model_name, target)

        # Create summary report
        summary = self.evaluator.create_summary_report(results)
        print(summary)

        logger.info("Evaluation complete")
        return results

    def predict(
        self,
        player_id: Optional[str] = None,
        player_data: Optional[pd.DataFrame] = None,
        use_ensemble: bool = True,
    ) -> Dict:
        """
        Generate projections for a player

        Args:
            player_id: Player ID to generate projections for
            player_data: Pre-processed player data
            use_ensemble: Whether to use ensemble predictions

        Returns:
            Dictionary of projections
        """
        logger.info(f"Generating projections for player: {player_id}")

        if player_data is None and player_id is None:
            raise ValueError("Must provide either player_id or player_data")

        if player_data is None:
            # Get player data from features
            player_data = self.features[self.features["player_id"] == player_id]

            if len(player_data) == 0:
                raise ValueError(f"Player {player_id} not found in data")

        # Prepare features
        target_cols = self.config["projections"].get(
            "batting_targets", []
        ) + self.config["projections"].get("pitching_targets", [])
        exclude_cols = target_cols + ["player_id", "cleaned_text", "combined_reports"]
        feature_cols = [
            col
            for col in player_data.columns
            if col not in exclude_cols
            and player_data[col].dtype in ["int64", "float64"]
        ]

        X = player_data[feature_cols]

        # Generate predictions
        projections = {}

        if use_ensemble and self.ensembles:
            for target, ensemble in self.ensembles.items():
                pred, uncertainty = ensemble.predict_with_uncertainty(X)
                projections[target] = {
                    "prediction": float(pred[0]),
                    "uncertainty": float(uncertainty[0]),
                }
        else:
            for target, target_models in self.models.items():
                # Use best model (first available)
                model = list(target_models.values())[0]
                pred = model.predict(X)
                projections[target] = {"prediction": float(pred[0])}

        logger.info(f"Generated projections for {len(projections)} targets")
        return projections

    def save_models(self, save_dir: Optional[str] = None) -> None:
        """
        Save trained models

        Args:
            save_dir: Directory to save models
        """
        save_dir = save_dir or self.config["data"]["models_dir"]
        self.model_trainer.save_models(save_dir)
        logger.info(f"Models saved to {save_dir}")

    def run_full_pipeline(
        self, stats_file: Optional[str] = None, reports_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run complete pipeline from data loading to evaluation

        Args:
            stats_file: CSV file with prospect statistics
            reports_file: CSV/JSON file with scouting reports

        Returns:
            DataFrame with evaluation results
        """
        logger.info("Running full projection pipeline")

        # Load data
        self.load_data(stats_file, reports_file)

        # Process NLP
        self.process_nlp()

        # Engineer features
        self.engineer_features()

        # Prepare training data
        X_train, X_test, y_train, y_test = self.prepare_training_data()

        # Train models
        self.train(X_train, y_train)

        # Evaluate
        results = self.evaluate(X_test, y_test)

        # Save models
        self.save_models()

        logger.info("Full pipeline complete")
        return results
