"""
"""Comprehensive example demonstrating all ScoutIQ features
"""

from pathlib import Path
import pandas as pd
import numpy as np
from src.pipeline import ProspectProjectionPipeline
from src.utils import setup_logger

# Constants
DATA_RAW = "data/raw"

# Initialize random generator
rng = np.random.default_rng(42)

logger = setup_logger("comprehensive_example")


def demonstrate_data_loading():
    """Demonstrate data loading capabilities"""
    print("\n" + "=" * 80)
    print("1. DATA LOADING")
    print("=" * 80)

    from src.data_ingestion import StructuredDataLoader, UnstructuredDataLoader

    # Load structured data
    struct_loader = StructuredDataLoader(DATA_RAW)
    stats_df = struct_loader.load_csv("prospect_stats.csv")
    print(f"\n✓ Loaded {len(stats_df)} player statistics")
    print(f"  Columns: {list(stats_df.columns[:10])}...")

    # Load scouting reports
    text_loader = UnstructuredDataLoader(DATA_RAW)
    reports_df = text_loader.load_scouting_reports("scouting_reports.csv")
    print(f"\n✓ Loaded {len(reports_df)} scouting reports")
    print("\n  Sample report:")
    print(f"  {reports_df.iloc[0]['report_text'][:200]}...")

    return stats_df, reports_df


def demonstrate_nlp_processing(reports_df):
    """Demonstrate NLP processing"""
    print("\n" + "=" * 80)
    print("2. NLP PROCESSING")
    print("=" * 80)

    from src.nlp import NLPPipeline

    nlp = NLPPipeline(use_spacy=False)

    # Process a single report
    sample_report = reports_df.iloc[0]["report_text"]
    features = nlp.process_report(sample_report)

    print("\n✓ Extracted NLP features:")
    print(f"  - Text length: {features['text_length']}")
    print(f"  - Word count: {features['word_count']}")
    print(f"  - Sentiment polarity: {features['sentiment_polarity']:.3f}")
    print(f"  - Sentiment class: {features['sentiment_class']}")

    if "power_grade" in features:
        print(f"  - Power grade: {features['power_grade']}")
    if "hit_grade" in features:
        print(f"  - Hit grade: {features['hit_grade']}")

    print(f"  - Strength mentions: {features['strength_mentions']}")
    print(f"  - Concern mentions: {features['concern_mentions']}")

    return features


def demonstrate_feature_engineering(stats_df):
    """Demonstrate feature engineering"""
    print("\n" + "=" * 80)
    print("3. FEATURE ENGINEERING")
    print("=" * 80)

    from src.features import FeaturePipeline

    feature_pipe = FeaturePipeline()

    # Show original features
    print(f"\n  Original features: {len(stats_df.columns)}")

    # Engineer features
    engineered_df = feature_pipe.engineer_features(
        stats_df.head(10),  # Use subset for demo
        create_interactions=True,
        scale_features=False,
    )

    print(f"  Engineered features: {len(engineered_df.columns)}")
    print(f"  New features added: {len(engineered_df.columns) - len(stats_df.columns)}")

    # Show some new features
    new_features = [col for col in engineered_df.columns if col not in stats_df.columns]
    print(f"\n  Sample new features: {new_features[:10]}")

    return engineered_df


def demonstrate_model_training():
    """Demonstrate model training"""
    print("\n" + "=" * 80)
    print("4. MODEL TRAINING")
    print("=" * 80)

    from src.models import RandomForestModel, XGBoostModel

    # Create synthetic training data for demo
    X_train = pd.DataFrame(
        rng.standard_normal((100, 10)), columns=[f"feature_{i}" for i in range(10)]
    )
    y_train = pd.Series(rng.standard_normal(100), name="target")
    X_test = pd.DataFrame(
        rng.standard_normal((20, 10)), columns=[f"feature_{i}" for i in range(10)]
    )

    # Train Random Forest
    print("\n  Training Random Forest...")
    rf_model = RandomForestModel(
        {"n_estimators": 50, "max_depth": 5, "random_state": 42}
    )
    rf_model.train(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    print(f"  ✓ Random Forest trained - Sample predictions: {rf_preds[:5]}")

    # Train XGBoost
    print("\n  Training XGBoost...")
    try:
        xgb_model = XGBoostModel(
            {"n_estimators": 50, "max_depth": 3, "random_state": 42}
        )
        xgb_model.train(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        print(f"  ✓ XGBoost trained - Sample predictions: {xgb_preds[:5]}")
    except ImportError:
        print("  ⚠ XGBoost not available (optional)")

    return rf_model


def demonstrate_full_pipeline():
    """Demonstrate the full projection pipeline"""
    print("\n" + "=" * 80)
    print("5. FULL PIPELINE EXECUTION")
    print("=" * 80)

    # Check if data exists
    data_dir = Path("data/raw")
    stats_file = data_dir / "prospect_stats.csv"
    reports_file = data_dir / "scouting_reports.csv"

    if not stats_file.exists() or not reports_file.exists():
        print("\n  ⚠ Sample data not found. Generating...")
        print("  Run: python scripts/generate_sample_data.py")
        return

    # Initialize and run pipeline
    print("\n  Initializing pipeline...")
    pipeline = ProspectProjectionPipeline()

    print("  Loading data...")
    pipeline.load_data()

    print("  Processing NLP...")
    pipeline.process_nlp()

    print("  Engineering features...")
    pipeline.engineer_features()

    print("  Preparing training data...")
    X_train, X_test, y_train, y_test = pipeline.prepare_training_data()
    print(f"    - Training samples: {len(X_train)}")
    print(f"    - Test samples: {len(X_test)}")
    print(f"    - Features: {len(X_train.columns)}")
    print(f"    - Targets: {list(y_train.columns)}")

    print("\n  Training models...")
    pipeline.train(X_train, y_train)
    print("  ✓ Models trained successfully")

    print("\n  Evaluating models...")
    results = pipeline.evaluate(X_test, y_test)

    print("\n  ✓ Pipeline complete!")
    print(f"\n  Best model performance by target:")
    for target in results["target"].unique():
        target_results = results[results["target"] == target]
        best_model = target_results.loc[target_results["mae"].idxmin()]
        print(f"    {target}: {best_model['model']} (MAE: {best_model['mae']:.4f})")

    # Make a prediction
    print("\n" + "=" * 80)
    print("6. PLAYER PROJECTION EXAMPLE")
    print("=" * 80)

    player_id = pipeline.features["player_id"].iloc[0]
    player_name = pipeline.data[pipeline.data["player_id"] == player_id]["name"].iloc[0]

    projections = pipeline.predict(player_id=player_id, use_ensemble=True)

    print(f\"\\n  Player: {player_name} (ID: {player_id})\")
    print(\"  Projections:\")
    for target, values in projections.items():
        pred = values["prediction"]
        unc = values.get("uncertainty", 0)
        print(f"    {target:15s}: {pred:6.3f} ± {unc:.3f}")

    # Save models
    print("\n  Saving models...")
    pipeline.save_models()
    print("  ✓ Models saved to data/models/")

    return pipeline


def main():
    """Run comprehensive demonstration"""
    print("\n" + "=" * 80)
    print("SCOUTIQ - COMPREHENSIVE EXAMPLE")
    print("Baseball Prospect Projection System")
    print("=" * 80)

    try:
        # Individual component demonstrations
        stats_df, reports_df = demonstrate_data_loading()
        _ = demonstrate_nlp_processing(reports_df)
        _ = demonstrate_feature_engineering(stats_df)
        _ = demonstrate_model_training()

        # Full pipeline
        _ = demonstrate_full_pipeline()

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Review the generated models in data/models/")
        print("  2. Check evaluation results in results/")
        print("  3. Customize config/config.yaml for your needs")
        print("  4. Replace sample data with your actual data")
        print("  5. Explore the Jupyter notebooks for interactive analysis")

    except Exception as e:
        logger.error(f"Error during demonstration: {str(e)}")
        import traceback

        traceback.print_exc()
        print(
            "\n⚠ If data files are missing, run: python scripts/generate_sample_data.py"
        )


if __name__ == "__main__":
    main()
