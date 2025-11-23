"""
Example: Using Lahman Baseball Database for player projections
"""

import pandas as pd
from src.data_ingestion import LahmanDataLoader
from src.features.engineering import StatisticalFeatureEngineer
from src.models.models import ModelTrainer, RandomForestModel, XGBoostModel
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import setup_logger
from sklearn.model_selection import train_test_split

logger = setup_logger("lahman_example")


def main():
    """Build a projection system using Lahman historical data"""

    print("=" * 80)
    print("LAHMAN BASEBALL DATABASE - PLAYER PROJECTION SYSTEM")
    print("=" * 80)

    # Step 1: Load historical data
    print("\n1. Loading Lahman Baseball Database...")
    loader = LahmanDataLoader()

    # Create player-season records (2000-2024, modern era)
    player_seasons = loader.create_player_seasons(
        min_year=2000, min_plate_appearances=200, include_pitchers=False
    )

    print(f"   ✓ Loaded {len(player_seasons)} player-seasons from 2000-2024")
    print(f"   ✓ Players: {player_seasons['playerID'].nunique()}")
    print(f"   ✓ Columns: {list(player_seasons.columns[:15])}...")

    # Step 2: Prepare projection data
    print("\n2. Preparing projection training data...")
    print("   (Using 3-year history to predict next season)")

    features, targets = loader.prepare_projection_data(
        current_year=2023, lookback_years=3, target_year_offset=1
    )

    if len(features) == 0:
        print("   ⚠️  No training data available")
        return

    print(f"   ✓ Created {len(features)} training samples")
    print(f"   ✓ Feature columns: {len(features.columns)}")
    print(f"   ✓ Target metrics: {list(targets.columns)}")

    # Step 3: Feature engineering
    print("\n3. Engineering additional features...")

    # Select numeric features for modeling
    feature_cols = [col for col in features.columns if col not in ["playerID"]]
    X = features[feature_cols].fillna(0)

    # Align targets with features
    y = targets.set_index(targets.index).drop(columns=["playerID"], errors="ignore")

    print(f"   ✓ Feature matrix: {X.shape}")
    print(f"   ✓ Target matrix: {y.shape}")

    # Step 4: Split data
    print("\n4. Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   ✓ Training: {len(X_train)} samples")
    print(f"   ✓ Testing: {len(X_test)} samples")

    # Step 5: Train models
    print("\n5. Training projection models...")

    # Train a Random Forest for each target metric
    target_metrics = ["AVG", "OBP", "SLG", "HR"]
    trained_models = {}

    for metric in target_metrics:
        if metric not in y_train.columns:
            continue

        print(f"\n   Training {metric} projection model...")

        # Train Random Forest
        rf_model = RandomForestModel(
            config={
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "max_features": "sqrt",
                "random_state": 42,
            }
        )

        rf_model.train(X_train, y_train[metric], task="regression")
        trained_models[metric] = rf_model

        # Make predictions
        train_pred = rf_model.predict(X_train)
        test_pred = rf_model.predict(X_test)

        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, r2_score

        train_mae = mean_absolute_error(y_train[metric], train_pred)
        test_mae = mean_absolute_error(y_test[metric], test_pred)
        test_r2 = r2_score(y_test[metric], test_pred)

        print(f"      Train MAE: {train_mae:.4f}")
        print(f"      Test MAE:  {test_mae:.4f}")
        print(f"      Test R²:   {test_r2:.4f}")

    # Step 6: Feature importance
    print("\n6. Top feature importance (AVG projection):")
    if "AVG" in trained_models:
        importance = trained_models["AVG"].get_feature_importance().head(10)
        for feat, imp in importance.items():
            print(f"   {feat:30s}: {imp:.4f}")

    # Step 7: Make sample projections
    print("\n7. Sample projections for test players:")
    sample_indices = X_test.head(5).index

    for idx in sample_indices:
        print(f"\n   Player {idx}:")
        print(f"      Historical AVG: {X_test.loc[idx, 'AVG_mean']:.3f}")
        print(f"      Actual 2024:    {y_test.loc[idx, 'AVG']:.3f}")
        if "AVG" in trained_models:
            pred = trained_models["AVG"].predict(X_test.loc[[idx]])[0]
            print(f"      Predicted:      {pred:.3f}")
            print(f"      Error:          {abs(pred - y_test.loc[idx, 'AVG']):.3f}")

    # Step 8: Save models
    print("\n8. Saving models...")
    import joblib
    from pathlib import Path

    model_dir = Path("data/models/lahman_projections")
    model_dir.mkdir(parents=True, exist_ok=True)

    for metric, model in trained_models.items():
        model_path = model_dir / f"{metric.lower()}_projection.pkl"
        joblib.dump(model, model_path)
        print(f"   ✓ Saved {metric} model to {model_path}")

    print("\n" + "=" * 80)
    print("PROJECTION SYSTEM COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Models saved to data/models/lahman_projections/")
    print("  2. Add more features (career trends, park factors, etc.)")
    print("  3. Incorporate recent minor league data for prospects")
    print("  4. Build ensemble models combining multiple algorithms")
    print("  5. Add uncertainty estimates to projections")


if __name__ == "__main__":
    main()
