"""
Feature engineering module for creating ML features from structured and unstructured data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class StatisticalFeatureEngineer:
    """
    Create features from structured statistical data
    """

    def __init__(self):
        """Initialize statistical feature engineer"""
        self.scalers = {}
        logger.info("Initialized StatisticalFeatureEngineer")

    def create_rate_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rate statistics from counting stats

        Args:
            df: DataFrame with baseball statistics

        Returns:
            DataFrame with added rate stats
        """
        logger.info("Creating rate statistics")

        df = df.copy()

        # Batting rate stats
        if "pa" in df.columns and "ab" in df.columns:
            df["bb_rate"] = df.get("bb", 0) / df["pa"].replace(0, 1) * 100
            df["k_rate"] = df.get("k", 0) / df["pa"].replace(0, 1) * 100
            df["iso"] = df.get("slg", 0) - df.get("avg", 0)

            if "h" in df.columns and "hr" in df.columns:
                df["babip"] = (df["h"] - df["hr"]) / (
                    df["ab"] - df["k"] - df["hr"] + df.get("sf", 0)
                ).replace(0, 1)

        # Pitching rate stats
        if "ip" in df.columns:
            df["k_per_9"] = df.get("k", 0) / df["ip"].replace(0, 1) * 9
            df["bb_per_9"] = df.get("bb", 0) / df["ip"].replace(0, 1) * 9
            df["hr_per_9"] = df.get("hr", 0) / df["ip"].replace(0, 1) * 9

            if "k" in df.columns and "bb" in df.columns:
                df["k_bb_ratio"] = df["k"] / df["bb"].replace(0, 1)

        return df

    def create_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced baseball metrics

        Args:
            df: DataFrame with baseball statistics

        Returns:
            DataFrame with advanced metrics
        """
        logger.info("Creating advanced metrics")

        df = df.copy()

        # wOBA weights (approximate 2024 values)
        woba_weights = {
            "bb": 0.69,
            "hbp": 0.72,
            "1b": 0.88,
            "2b": 1.24,
            "3b": 1.56,
            "hr": 2.08,
        }

        # Calculate wOBA if we have the necessary stats
        if all(col in df.columns for col in ["bb", "h", "hr", "ab", "sf"]):
            df["1b"] = df["h"] - df.get("2b", 0) - df.get("3b", 0) - df["hr"]

            numerator = (
                woba_weights["bb"] * df.get("bb", 0)
                + woba_weights["hbp"] * df.get("hbp", 0)
                + woba_weights["1b"] * df["1b"]
                + woba_weights["2b"] * df.get("2b", 0)
                + woba_weights["3b"] * df.get("3b", 0)
                + woba_weights["hr"] * df["hr"]
            )

            denominator = (
                df["ab"]
                + df.get("bb", 0)
                - df.get("ibb", 0)
                + df["sf"]
                + df.get("hbp", 0)
            )
            df["woba"] = numerator / denominator.replace(0, 1)

        # Power metrics
        if "ab" in df.columns:
            df["hr_rate"] = df.get("hr", 0) / df["ab"].replace(0, 1)
            df["xbh_rate"] = (df.get("2b", 0) + df.get("3b", 0) + df.get("hr", 0)) / df[
                "ab"
            ].replace(0, 1)

        # Speed metrics
        if "sb" in df.columns and "cs" in df.columns:
            df["sb_pct"] = df["sb"] / (df["sb"] + df["cs"]).replace(0, 1)

        return df

    def create_age_adjusted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-adjusted performance features

        Args:
            df: DataFrame with age and performance stats

        Returns:
            DataFrame with age-adjusted features
        """
        logger.info("Creating age-adjusted features")

        df = df.copy()

        if "age" not in df.columns:
            logger.warning("Age column not found, skipping age adjustments")
            return df

        # Age relative to level
        level_age_map = {
            "rookie": 19,
            "A": 20,
            "A+": 21,
            "AA": 22,
            "AAA": 24,
            "MLB": 27,
        }

        if "level" in df.columns:
            df["expected_age"] = df["level"].map(level_age_map)
            df["age_vs_level"] = df["age"] - df["expected_age"]
            df["age_adjusted_ops"] = df.get("ops", 0) * (1 + df["age_vs_level"] * 0.02)

        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features (trends, recent performance)

        Args:
            df: DataFrame with temporal data

        Returns:
            DataFrame with temporal features
        """
        logger.info("Creating temporal features")

        df = df.copy()

        # Assume data is sorted by player and date
        if "player_id" in df.columns and "date" in df.columns:
            # Calculate rolling averages
            df = df.sort_values(["player_id", "date"])

            for stat in ["avg", "obp", "slg", "ops"]:
                if stat in df.columns:
                    df[f"{stat}_last_30"] = df.groupby("player_id")[stat].transform(
                        lambda x: x.rolling(window=30, min_periods=1).mean()
                    )
                    df[f"{stat}_trend"] = df.groupby("player_id")[stat].transform(
                        lambda x: x.diff()
                    )

        return df

    def create_interaction_features(
        self, df: pd.DataFrame, pairs: List[tuple]
    ) -> pd.DataFrame:
        """
        Create interaction features between variables

        Args:
            df: Input DataFrame
            pairs: List of column pairs to interact

        Returns:
            DataFrame with interaction features
        """
        logger.info(f"Creating {len(pairs)} interaction features")

        df = df.copy()

        for col1, col2 in pairs:
            if col1 in df.columns and col2 in df.columns:
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

        return df

    def scale_features(
        self, df: pd.DataFrame, columns: List[str], method: str = "standard"
    ) -> pd.DataFrame:
        """
        Scale numerical features

        Args:
            df: Input DataFrame
            columns: Columns to scale
            method: Scaling method ('standard' or 'minmax')

        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling {len(columns)} features using {method} method")

        df = df.copy()

        scaler = StandardScaler() if method == "standard" else MinMaxScaler()

        # Only scale columns that exist
        cols_to_scale = [c for c in columns if c in df.columns]

        if cols_to_scale:
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
            self.scalers[method] = scaler

        return df


class NLPFeatureEngineer:
    """
    Create features from NLP-processed text data
    """

    def __init__(self):
        """Initialize NLP feature engineer"""
        logger.info("Initialized NLPFeatureEngineer")

    def aggregate_tool_grades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate tool grades into summary features

        Args:
            df: DataFrame with tool grade columns

        Returns:
            DataFrame with aggregated features
        """
        logger.info("Aggregating tool grades")

        df = df.copy()

        # Position player tool columns
        position_tools = [
            col
            for col in df.columns
            if any(
                tool in col
                for tool in [
                    "hit_grade",
                    "power_grade",
                    "run_grade",
                    "arm_grade",
                    "field_grade",
                ]
            )
        ]

        if position_tools:
            df["avg_tool_grade"] = df[position_tools].mean(axis=1)
            df["max_tool_grade"] = df[position_tools].max(axis=1)
            df["min_tool_grade"] = df[position_tools].min(axis=1)
            df["tool_variance"] = df[position_tools].std(axis=1)

        # Pitcher tool columns
        pitcher_tools = [
            col
            for col in df.columns
            if any(
                tool in col
                for tool in [
                    "fastball_grade",
                    "curveball_grade",
                    "slider_grade",
                    "changeup_grade",
                ]
            )
        ]

        if pitcher_tools:
            df["avg_pitch_grade"] = df[pitcher_tools].mean(axis=1)
            df["max_pitch_grade"] = df[pitcher_tools].max(axis=1)
            df["num_plus_pitches"] = (df[pitcher_tools] >= 60).sum(axis=1)

        return df

    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from text characteristics

        Args:
            df: DataFrame with text data

        Returns:
            DataFrame with text features
        """
        logger.info("Creating text features")

        df = df.copy()

        if "cleaned_text" in df.columns:
            # Text length features
            df["text_length"] = df["cleaned_text"].str.len()
            df["word_count"] = df["cleaned_text"].str.split().str.len()
            df["avg_word_length"] = df["text_length"] / df["word_count"].replace(0, 1)

            # Sentence count
            df["sentence_count"] = df["cleaned_text"].str.count(r"[.!?]") + 1
            df["avg_sentence_length"] = df["word_count"] / df["sentence_count"]

        return df

    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived sentiment features

        Args:
            df: DataFrame with sentiment scores

        Returns:
            DataFrame with sentiment features
        """
        logger.info("Creating sentiment features")

        df = df.copy()

        if "sentiment_polarity" in df.columns:
            # Sentiment strength
            df["sentiment_strength"] = df["sentiment_polarity"].abs()

            # Weighted sentiment (by subjectivity)
            if "sentiment_subjectivity" in df.columns:
                df["weighted_sentiment"] = (
                    df["sentiment_polarity"] * df["sentiment_subjectivity"]
                )

        # Ratio of strengths to concerns
        if "strength_mentions" in df.columns and "concern_mentions" in df.columns:
            df["strength_concern_ratio"] = df["strength_mentions"] / (
                df["concern_mentions"] + 1
            )

        return df


class CompositeFeatureEngineer:
    """
    Create composite features combining structured and unstructured data
    """

    def __init__(self):
        """Initialize composite feature engineer"""
        logger.info("Initialized CompositeFeatureEngineer")

    def align_stats_and_grades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that align statistical performance with tool grades

        Args:
            df: DataFrame with both stats and grades

        Returns:
            DataFrame with alignment features
        """
        logger.info("Creating stat-grade alignment features")

        df = df.copy()

        # Power: HR rate vs power grade
        if "hr_rate" in df.columns and "power_grade" in df.columns:
            df["power_alignment"] = df["hr_rate"] * 100 - df["power_grade"]

        # Speed: SB vs run grade
        if "sb" in df.columns and "run_grade" in df.columns:
            df["speed_alignment"] = (df["sb"] / 20) * 50 - df["run_grade"]

        # Hit tool: AVG vs hit grade
        if "avg" in df.columns and "hit_grade" in df.columns:
            df["hit_alignment"] = (df["avg"] - 0.200) * 250 - df["hit_grade"]

        return df

    def create_projection_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weights for blending different data sources

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with projection weights
        """
        logger.info("Creating projection weights")

        df = df.copy()

        # Weight based on sample size
        if "pa" in df.columns:
            df["stat_reliability"] = np.minimum(df["pa"] / 500, 1.0)

        # Weight based on number of reports
        if "report_count" in df.columns:
            df["report_reliability"] = np.minimum(df["report_count"] / 5, 1.0)

        # Combined reliability
        if "stat_reliability" in df.columns and "report_reliability" in df.columns:
            df["overall_reliability"] = (
                df["stat_reliability"] + df["report_reliability"]
            ) / 2

        return df

    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features indicating risk factors

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with risk features
        """
        logger.info("Creating risk features")

        df = df.copy()

        # Age risk (older prospects have less upside)
        if "age" in df.columns:
            df["age_risk"] = np.maximum(0, df["age"] - 23) / 5

        # Level risk (lower levels = more projection needed)
        level_risk_map = {
            "rookie": 1.0,
            "A": 0.8,
            "A+": 0.6,
            "AA": 0.4,
            "AAA": 0.2,
            "MLB": 0.0,
        }

        if "level" in df.columns:
            df["level_risk"] = df["level"].map(level_risk_map)

        # Injury risk from text
        if "concern_mentions" in df.columns:
            df["injury_risk"] = df["concern_mentions"] / 10

        # Overall risk score
        risk_cols = [
            col for col in df.columns if "risk" in col and col != "overall_risk"
        ]
        if risk_cols:
            df["overall_risk"] = df[risk_cols].mean(axis=1)

        return df


class FeaturePipeline:
    """
    Complete feature engineering pipeline
    """

    def __init__(self):
        """Initialize feature pipeline"""
        self.stat_engineer = StatisticalFeatureEngineer()
        self.nlp_engineer = NLPFeatureEngineer()
        self.composite_engineer = CompositeFeatureEngineer()
        logger.info("Initialized FeaturePipeline")

    def engineer_features(
        self,
        df: pd.DataFrame,
        create_interactions: bool = True,
        scale_features: bool = False,
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps

        Args:
            df: Input DataFrame
            create_interactions: Whether to create interaction features
            scale_features: Whether to scale numerical features

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline")

        df = df.copy()

        # Statistical features
        df = self.stat_engineer.create_rate_stats(df)
        df = self.stat_engineer.create_advanced_metrics(df)
        df = self.stat_engineer.create_age_adjusted_features(df)
        df = self.stat_engineer.create_temporal_features(df)

        # NLP features
        df = self.nlp_engineer.aggregate_tool_grades(df)
        df = self.nlp_engineer.create_text_features(df)
        df = self.nlp_engineer.create_sentiment_features(df)

        # Composite features
        df = self.composite_engineer.align_stats_and_grades(df)
        df = self.composite_engineer.create_projection_weights(df)
        df = self.composite_engineer.create_risk_features(df)

        # Interaction features
        if create_interactions:
            interaction_pairs = [
                ("age", "level_risk"),
                ("avg_tool_grade", "stat_reliability"),
                ("sentiment_polarity", "report_count"),
            ]
            df = self.stat_engineer.create_interaction_features(df, interaction_pairs)

        # Feature scaling
        if scale_features:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude ID columns and target variables
            cols_to_scale = [
                c
                for c in numeric_cols
                if not any(x in c.lower() for x in ["id", "mlb_", "target"])
            ]
            df = self.stat_engineer.scale_features(df, cols_to_scale)

        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groups for analysis

        Returns:
            Dictionary of feature groups
        """
        return {
            "statistical": [
                "avg",
                "obp",
                "slg",
                "ops",
                "woba",
                "iso",
                "bb_rate",
                "k_rate",
            ],
            "tool_grades": [
                "hit_grade",
                "power_grade",
                "run_grade",
                "arm_grade",
                "field_grade",
            ],
            "sentiment": [
                "sentiment_polarity",
                "sentiment_subjectivity",
                "strength_concern_ratio",
            ],
            "age_level": ["age", "age_vs_level", "level_risk"],
            "risk": ["age_risk", "level_risk", "injury_risk", "overall_risk"],
            "composite": ["power_alignment", "speed_alignment", "hit_alignment"],
        }
