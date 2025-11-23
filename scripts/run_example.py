"""
Example script demonstrating the projection pipeline
"""

from src.pipeline import ProspectProjectionPipeline
from src.utils import setup_logger

logger = setup_logger(__name__)


def main():
    """Run example pipeline"""

    print("=" * 80)
    print("Baseball Prospect Projection System - Example")
    print("=" * 80)
    print()

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = ProspectProjectionPipeline(config_path="config/config.yaml")

    # Run full pipeline
    print("\nRunning full pipeline...")
    try:
        results = pipeline.run_full_pipeline(
            stats_file="prospect_stats.csv", reports_file="scouting_reports.csv"
        )

        print("\n" + "=" * 80)
        print("Pipeline Complete!")
        print("=" * 80)
        print("\nEvaluation Results:")
        print(results.to_string())

        # Example prediction for a single player
        print("\n" + "=" * 80)
        print("Example Player Projection")
        print("=" * 80)

        player_id = pipeline.features["player_id"].iloc[0]
        projections = pipeline.predict(player_id=player_id, use_ensemble=True)

        print(f"\nProjections for Player: {player_id}")
        for target, values in projections.items():
            pred = values["prediction"]
            unc = values.get("uncertainty", 0)
            print(f"  {target}: {pred:.3f} ± {unc:.3f}")

    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
