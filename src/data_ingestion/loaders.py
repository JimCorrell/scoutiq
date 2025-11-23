"""
Data ingestion module for loading structured and unstructured baseball prospect data
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class StructuredDataLoader:
    """
    Loader for structured statistical data (CSV, JSON, Parquet)
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize structured data loader

        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized StructuredDataLoader with directory: {self.data_dir}")

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file

        Args:
            filename: Name of CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame with loaded data
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading CSV file: {filepath}")

        try:
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"Loaded {len(df)} rows from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {filename}: {str(e)}")
            raise

    def load_json(self, filename: str) -> pd.DataFrame:
        """
        Load data from JSON file

        Args:
            filename: Name of JSON file

        Returns:
            DataFrame with loaded data
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading JSON file: {filepath}")

        try:
            df = pd.read_json(filepath)
            logger.info(f"Loaded {len(df)} rows from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading JSON file {filename}: {str(e)}")
            raise

    def load_parquet(self, filename: str) -> pd.DataFrame:
        """
        Load data from Parquet file

        Args:
            filename: Name of Parquet file

        Returns:
            DataFrame with loaded data
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading Parquet file: {filepath}")

        try:
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded {len(df)} rows from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading Parquet file {filename}: {str(e)}")
            raise

    def validate_required_columns(
        self, df: pd.DataFrame, required_cols: List[str]
    ) -> bool:
        """
        Validate that DataFrame contains required columns

        Args:
            df: DataFrame to validate
            required_cols: List of required column names

        Returns:
            True if all required columns present
        """
        missing_cols = set(required_cols) - set(df.columns)

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False

        logger.info("All required columns present")
        return True

    def clean_structured_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning for structured data

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning structured data")

        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)

        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} duplicate rows")

        # Convert data types
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    logger.info(f"Converted {col} to datetime")
                except:
                    pass

        return df


class UnstructuredDataLoader:
    """
    Loader for unstructured text data (scouting reports, player descriptions)
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize unstructured data loader

        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Initialized UnstructuredDataLoader with directory: {self.data_dir}"
        )

    def load_scouting_reports(self, filename: str) -> pd.DataFrame:
        """
        Load scouting reports from CSV or JSON

        Args:
            filename: Name of file containing scouting reports

        Returns:
            DataFrame with columns: player_id, report_date, scout_name, report_text
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading scouting reports from: {filepath}")

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(filepath)
            elif filename.endswith(".json"):
                df = pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filename}")

            logger.info(f"Loaded {len(df)} scouting reports")
            return df
        except Exception as e:
            logger.error(f"Error loading scouting reports: {str(e)}")
            raise

    def load_text_files(
        self, directory: str, player_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Load individual text files (one per player/report)

        Args:
            directory: Directory containing text files
            player_mapping: Optional mapping from filename to player_id

        Returns:
            DataFrame with player_id and report_text
        """
        text_dir = self.data_dir / directory
        logger.info(f"Loading text files from: {text_dir}")

        if not text_dir.exists():
            raise FileNotFoundError(f"Directory not found: {text_dir}")

        reports = []

        for file_path in text_dir.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                filename = file_path.stem
                player_id = (
                    player_mapping.get(filename, filename)
                    if player_mapping
                    else filename
                )

                reports.append(
                    {
                        "player_id": player_id,
                        "filename": filename,
                        "report_text": text,
                        "file_date": datetime.fromtimestamp(file_path.stat().st_mtime),
                    }
                )

            except Exception as e:
                logger.warning(f"Error reading {file_path}: {str(e)}")

        df = pd.DataFrame(reports)
        logger.info(f"Loaded {len(df)} text files")
        return df

    def validate_text_data(
        self, df: pd.DataFrame, text_col: str = "report_text"
    ) -> pd.DataFrame:
        """
        Validate and clean text data

        Args:
            df: DataFrame containing text data
            text_col: Name of column containing text

        Returns:
            Validated DataFrame
        """
        logger.info("Validating text data")

        initial_rows = len(df)

        # Remove rows with missing text
        df = df[df[text_col].notna()]
        df = df[df[text_col].str.strip() != ""]

        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} rows with missing/empty text")

        # Add text length column
        df["text_length"] = df[text_col].str.len()

        logger.info(
            f"Text length - Mean: {df['text_length'].mean():.0f}, "
            f"Min: {df['text_length'].min()}, Max: {df['text_length'].max()}"
        )

        return df


class DataIntegrator:
    """
    Integrate structured and unstructured data sources
    """

    def __init__(self):
        """Initialize data integrator"""
        logger.info("Initialized DataIntegrator")

    def merge_data(
        self,
        structured_df: pd.DataFrame,
        unstructured_df: pd.DataFrame,
        on: str = "player_id",
        how: str = "left",
    ) -> pd.DataFrame:
        """
        Merge structured and unstructured data

        Args:
            structured_df: DataFrame with structured stats
            unstructured_df: DataFrame with text reports
            on: Column to merge on
            how: Type of merge (left, right, inner, outer)

        Returns:
            Merged DataFrame
        """
        logger.info(f"Merging data on column '{on}' using '{how}' join")

        # Aggregate multiple reports per player
        if len(unstructured_df) > 0 and on in unstructured_df.columns:
            # Combine all reports for each player
            aggregated_reports = (
                unstructured_df.groupby(on)
                .agg(
                    {
                        "report_text": lambda x: " [REPORT_SEP] ".join(x),
                        "report_date": (
                            "max"
                            if "report_date" in unstructured_df.columns
                            else lambda x: None
                        ),
                    }
                )
                .reset_index()
            )

            aggregated_reports.rename(
                columns={"report_text": "combined_reports"}, inplace=True
            )

            # Count reports per player
            report_counts = (
                unstructured_df.groupby(on).size().reset_index(name="report_count")
            )
            aggregated_reports = aggregated_reports.merge(report_counts, on=on)

            # Merge with structured data
            merged_df = structured_df.merge(aggregated_reports, on=on, how=how)
        else:
            merged_df = structured_df

        logger.info(f"Merged data shape: {merged_df.shape}")
        return merged_df

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_col: Optional[str] = None,
    ) -> tuple:
        """
        Split data into train and test sets

        Args:
            df: Input DataFrame
            test_size: Proportion of data for test set
            random_state: Random seed
            stratify_col: Column to stratify split on

        Returns:
            Tuple of (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split

        logger.info(
            f"Splitting data: test_size={test_size}, random_state={random_state}"
        )

        stratify = (
            df[stratify_col] if stratify_col and stratify_col in df.columns else None
        )

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=stratify
        )

        logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
        return train_df, test_df
