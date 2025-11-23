import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion import StructuredDataLoader, UnstructuredDataLoader
from src.nlp import NLPPipeline, ToolGradeExtractor
from src.features import FeaturePipeline


class TestDataIngestion(unittest.TestCase):
    """Test data ingestion modules"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_data_dir = Path(__file__).parent / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)

        # Create test CSV
        test_df = pd.DataFrame(
            {
                "player_id": ["P001", "P002"],
                "name": ["Player 1", "Player 2"],
                "age": [21, 22],
                "avg": [0.285, 0.305],
            }
        )
        test_df.to_csv(self.test_data_dir / "test_stats.csv", index=False)

    def test_structured_loader(self):
        """Test structured data loading"""
        loader = StructuredDataLoader(str(self.test_data_dir))
        df = loader.load_csv("test_stats.csv")

        self.assertEqual(len(df), 2)
        self.assertIn("player_id", df.columns)

    def tearDown(self):
        """Clean up test files"""
        import shutil

        if self.test_data_dir.exists():
            shutil.rmtree(self.test_data_dir)


class TestNLPProcessing(unittest.TestCase):
    """Test NLP processing modules"""

    def test_tool_grade_extraction(self):
        """Test tool grade extraction"""
        extractor = ToolGradeExtractor()

        text = "Shows plus power with 60-grade raw power. Hit tool is above-average."
        grades = extractor.extract_grades(text)

        self.assertIn("power_grade", grades)
        self.assertEqual(grades["power_grade"], 60)

    def test_nlp_pipeline(self):
        """Test complete NLP pipeline"""
        pipeline = NLPPipeline(use_spacy=False)

        report = (
            "21-year-old SS at AA. Displays plus hit tool with good bat-to-ball skills."
        )
        features = pipeline.process_report(report)

        self.assertIn("sentiment_polarity", features)
        self.assertIn("text_length", features)
        self.assertGreater(features["text_length"], 0)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering modules"""

    def test_feature_pipeline(self):
        """Test feature engineering pipeline"""
        pipeline = FeaturePipeline()

        # Create test data
        df = pd.DataFrame(
            {
                "player_id": ["P001"],
                "age": [21],
                "level": ["AA"],
                "pa": [450],
                "ab": [400],
                "h": [114],
                "hr": [15],
                "bb": [45],
                "k": [90],
            }
        )

        # Engineer features
        result = pipeline.engineer_features(df, create_interactions=False)

        # Check that new features were created
        self.assertGreater(len(result.columns), len(df.columns))


if __name__ == "__main__":
    unittest.main()
