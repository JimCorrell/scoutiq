"""
Lahman Baseball Database loader for historical statistics
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LahmanDataLoader:
    """
    Load and process Lahman Baseball Database (1871-2024)
    """

    def __init__(self, lahman_dir: str = "data/lahman/lahman_1871-2024u_csv"):
        """
        Initialize Lahman data loader

        Args:
            lahman_dir: Directory containing Lahman CSV files
        """
        self.lahman_dir = Path(lahman_dir)
        if not self.lahman_dir.exists():
            raise FileNotFoundError(f"Lahman directory not found: {lahman_dir}")

        logger.info(f"Initialized Lahman loader from {lahman_dir}")

    def load_batting(self, min_year: Optional[int] = None) -> pd.DataFrame:
        """
        Load batting statistics

        Args:
            min_year: Minimum year to include (e.g., 2000 for modern era)

        Returns:
            DataFrame with batting statistics
        """
        file_path = self.lahman_dir / "Batting.csv"
        logger.info(f"Loading batting data from {file_path}")

        df = pd.read_csv(file_path)

        if min_year:
            df = df[df["yearID"] >= min_year]
            logger.info(f"Filtered to {min_year}+: {len(df)} records")

        return df

    def load_pitching(self, min_year: Optional[int] = None) -> pd.DataFrame:
        """
        Load pitching statistics

        Args:
            min_year: Minimum year to include

        Returns:
            DataFrame with pitching statistics
        """
        file_path = self.lahman_dir / "Pitching.csv"
        logger.info(f"Loading pitching data from {file_path}")

        df = pd.read_csv(file_path)

        if min_year:
            df = df[df["yearID"] >= min_year]
            logger.info(f"Filtered to {min_year}+: {len(df)} records")

        return df

    def load_people(self) -> pd.DataFrame:
        """
        Load player biographical data

        Returns:
            DataFrame with player information
        """
        file_path = self.lahman_dir / "People.csv"
        logger.info(f"Loading people data from {file_path}")

        df = pd.read_csv(file_path)

        # Calculate age-related fields
        df["birthDate"] = pd.to_datetime(
            df[["birthYear", "birthMonth", "birthDay"]].rename(
                columns={"birthYear": "year", "birthMonth": "month", "birthDay": "day"}
            ),
            errors="coerce",
        )

        return df

    def load_fielding(self, min_year: Optional[int] = None) -> pd.DataFrame:
        """
        Load fielding statistics

        Args:
            min_year: Minimum year to include

        Returns:
            DataFrame with fielding statistics
        """
        file_path = self.lahman_dir / "Fielding.csv"
        logger.info(f"Loading fielding data from {file_path}")

        df = pd.read_csv(file_path)

        if min_year:
            df = df[df["yearID"] >= min_year]

        return df

    def load_teams(self, min_year: Optional[int] = None) -> pd.DataFrame:
        """
        Load team statistics

        Args:
            min_year: Minimum year to include

        Returns:
            DataFrame with team statistics
        """
        file_path = self.lahman_dir / "Teams.csv"
        logger.info(f"Loading team data from {file_path}")

        df = pd.read_csv(file_path)

        if min_year:
            df = df[df["yearID"] >= min_year]

        return df

    def create_player_seasons(
        self,
        min_year: int = 2000,
        min_plate_appearances: int = 100,
        include_pitchers: bool = False,
    ) -> pd.DataFrame:
        """
        Create comprehensive player-season records for modeling

        Args:
            min_year: Minimum year to include (default 2000 for modern era)
            min_plate_appearances: Minimum PA to be included
            include_pitchers: Whether to include pitcher batting stats

        Returns:
            DataFrame with player-season records ready for feature engineering
        """
        logger.info(
            f"Creating player seasons: min_year={min_year}, min_pa={min_plate_appearances}"
        )

        # Load core datasets
        batting = self.load_batting(min_year=min_year)
        people = self.load_people()
        fielding = self.load_fielding(min_year=min_year)

        # Calculate plate appearances
        batting["PA"] = (
            batting["AB"]
            + batting["BB"].fillna(0)
            + batting["HBP"].fillna(0)
            + batting["SH"].fillna(0)
            + batting["SF"].fillna(0)
        )

        # Filter by minimum PA
        batting = batting[batting["PA"] >= min_plate_appearances].copy()

        # Merge with player biographical data
        df = batting.merge(
            people[
                [
                    "playerID",
                    "nameFirst",
                    "nameLast",
                    "birthYear",
                    "birthDate",
                    "weight",
                    "height",
                    "bats",
                    "throws",
                ]
            ],
            on="playerID",
            how="left",
        )

        # Calculate age during season
        df["age"] = df["yearID"] - df["birthYear"]

        # Add primary position from fielding data
        # Get the position where player had most games each season
        fielding_agg = (
            fielding.groupby(["playerID", "yearID", "POS"])["G"]
            .sum()
            .reset_index()
            .sort_values("G", ascending=False)
            .groupby(["playerID", "yearID"])
            .first()
            .reset_index()
        )

        df = df.merge(
            fielding_agg[["playerID", "yearID", "POS"]],
            on=["playerID", "yearID"],
            how="left",
        )

        # Calculate rate stats
        df["AVG"] = df["H"] / df["AB"].replace(0, 1)
        df["OBP"] = (df["H"] + df["BB"] + df["HBP"]) / df["PA"].replace(0, 1)
        df["SLG"] = (df["H"] + df["2B"] + 2 * df["3B"] + 3 * df["HR"]) / df[
            "AB"
        ].replace(0, 1)
        df["OPS"] = df["OBP"] + df["SLG"]

        # Fill missing values
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        logger.info(f"Created {len(df)} player-season records from {min_year}-2024")

        return df

    def create_career_trajectories(
        self, min_seasons: int = 3, min_year: int = 2000
    ) -> Dict[str, pd.DataFrame]:
        """
        Create career trajectory data for modeling player development

        Args:
            min_seasons: Minimum number of seasons required
            min_year: Starting year for analysis

        Returns:
            Dictionary with player career data
        """
        logger.info("Creating career trajectories")

        player_seasons = self.create_player_seasons(
            min_year=min_year, min_plate_appearances=100
        )

        # Calculate career stats by service time
        player_seasons = player_seasons.sort_values(["playerID", "yearID"])
        player_seasons["service_year"] = (
            player_seasons.groupby("playerID").cumcount() + 1
        )

        # Filter to players with minimum seasons
        player_counts = (
            player_seasons.groupby("playerID")["yearID"].count().reset_index()
        )
        qualified_players = player_counts[player_counts["yearID"] >= min_seasons][
            "playerID"
        ]

        trajectories = player_seasons[
            player_seasons["playerID"].isin(qualified_players)
        ].copy()

        logger.info(
            f"Created trajectories for {len(qualified_players)} players with {min_seasons}+ seasons"
        )

        return {
            "trajectories": trajectories,
            "player_counts": player_counts,
            "qualified_players": qualified_players.tolist(),
        }

    def prepare_projection_data(
        self,
        current_year: int = 2024,
        lookback_years: int = 5,
        target_year_offset: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for building projection models

        Args:
            current_year: Most recent year of data
            lookback_years: Number of years to use as features
            target_year_offset: Years ahead to predict (1 = next season)

        Returns:
            Tuple of (features DataFrame, targets DataFrame)
        """
        logger.info(
            f"Preparing projection data: {lookback_years}yr lookback, "
            f"{target_year_offset}yr ahead"
        )

        all_seasons = self.create_player_seasons(
            min_year=current_year - lookback_years - target_year_offset
        )

        features_list = []
        targets_list = []

        for year in range(
            current_year - lookback_years - target_year_offset + 1,
            current_year - target_year_offset + 1,
        ):
            # Get historical years as features
            historical = all_seasons[
                (all_seasons["yearID"] >= year - lookback_years)
                & (all_seasons["yearID"] < year)
            ]

            # Get next year as target
            targets = all_seasons[all_seasons["yearID"] == year + target_year_offset]

            # Group historical data by player
            hist_agg = (
                historical.groupby("playerID")
                .agg(
                    {
                        "yearID": "max",  # Most recent year in history
                        "age": "max",
                        "G": ["sum", "mean"],
                        "AB": ["sum", "mean"],
                        "H": ["sum", "mean"],
                        "2B": ["sum", "mean"],
                        "3B": ["sum", "mean"],
                        "HR": ["sum", "mean"],
                        "RBI": ["sum", "mean"],
                        "SB": ["sum", "mean"],
                        "BB": ["sum", "mean"],
                        "SO": ["sum", "mean"],
                        "AVG": ["mean", "std"],
                        "OBP": ["mean", "std"],
                        "SLG": ["mean", "std"],
                        "OPS": ["mean", "std"],
                    }
                )
                .reset_index()
            )

            # Flatten column names
            hist_agg.columns = [
                "_".join(col).strip("_") if col[1] else col[0]
                for col in hist_agg.columns
            ]

            # Merge with targets
            merged = hist_agg.merge(
                targets[
                    [
                        "playerID",
                        "yearID",
                        "AVG",
                        "OBP",
                        "SLG",
                        "OPS",
                        "HR",
                        "RBI",
                        "SB",
                    ]
                ],
                on="playerID",
                how="inner",
                suffixes=("_hist", "_target"),
            )

            if len(merged) > 0:
                # Extract features (all hist columns) and targets
                target_cols = ["AVG", "OBP", "SLG", "OPS", "HR", "RBI", "SB"]
                # Drop target columns and any yearID_target if it exists
                cols_to_drop = target_cols + [
                    col for col in merged.columns if col.endswith("_target")
                ]
                feature_cols = [
                    col for col in merged.columns if col not in cols_to_drop
                ]

                features_list.append(merged[feature_cols])
                targets_list.append(merged[["playerID"] + target_cols])

        if features_list:
            features = pd.concat(features_list, ignore_index=True)
            targets = pd.concat(targets_list, ignore_index=True)

            logger.info(f"Prepared {len(features)} training samples")

            return features, targets
        else:
            logger.warning("No training samples created")
            return pd.DataFrame(), pd.DataFrame()
