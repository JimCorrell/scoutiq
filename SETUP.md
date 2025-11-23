# ScoutIQ Setup Guide

## Quick Start

### 1. Install Dependencies

First, generate sample data (you'll need this for testing):

```bash
python scripts/generate_sample_data.py
```

### 2. Download NLP Models

```bash
# Download spaCy model
python -m spacy download en_core_web_lg
```

### 3. Run Example Pipeline

```bash
python scripts/run_example.py
```

## Project Structure

```
scoutiq/
├── config/
│   └── config.yaml              # Configuration settings
├── data/
│   ├── raw/                     # Raw data (CSV files)
│   ├── processed/               # Processed data
│   └── models/                  # Saved models
├── src/
│   ├── data_ingestion/          # Data loading modules
│   ├── nlp/                     # NLP processing
│   ├── features/                # Feature engineering
│   ├── models/                  # ML models
│   ├── evaluation/              # Model evaluation
│   └── pipeline.py              # Main pipeline
├── scripts/
│   ├── generate_sample_data.py  # Generate test data
│   └── run_example.py           # Example usage
├── notebooks/                   # Jupyter notebooks
└── requirements.txt             # Python dependencies
```

## Usage Examples

### Using the Pipeline

```python
from src.pipeline import ProspectProjectionPipeline

# Initialize pipeline
pipeline = ProspectProjectionPipeline()

# Run full pipeline
results = pipeline.run_full_pipeline(
    stats_file="prospect_stats.csv",
    reports_file="scouting_reports.csv"
)

# Make projection for a specific player
projections = pipeline.predict(player_id="P0001", use_ensemble=True)
print(projections)
```

### Loading Data

```python
from src.data_ingestion import StructuredDataLoader, UnstructuredDataLoader

# Load structured stats
loader = StructuredDataLoader("data/raw")
stats_df = loader.load_csv("prospect_stats.csv")

# Load scouting reports
reports_loader = UnstructuredDataLoader("data/raw")
reports_df = reports_loader.load_scouting_reports("scouting_reports.csv")
```

### NLP Processing

```python
from src.nlp import NLPPipeline

# Initialize NLP pipeline
nlp = NLPPipeline()

# Process a single report
report_text = "Plus bat speed with raw power potential. Shows good plate discipline..."
features = nlp.process_report(report_text)
print(features)
```

### Feature Engineering

```python
from src.features import FeaturePipeline

# Initialize feature pipeline
feature_pipe = FeaturePipeline()

# Engineer features
engineered_df = feature_pipe.engineer_features(data_df)
```

### Training Models

```python
from src.models import ModelTrainer, RandomForestModel, XGBoostModel

# Train a single model
rf_model = RandomForestModel()
rf_model.train(X_train, y_train)
predictions = rf_model.predict(X_test)

# Or use the trainer for multiple models
trainer = ModelTrainer(config)
models = trainer.train_models(X_train, y_train, X_val, y_val)
```

## Data Format

### Structured Data (prospect_stats.csv)

Required columns:

- `player_id`: Unique identifier
- `name`: Player name
- `age`: Age
- `level`: Minor league level (A, A+, AA, AAA)
- `pa`: Plate appearances
- `avg`: Batting average
- `obp`: On-base percentage
- `slg`: Slugging percentage
- `hr`: Home runs
- `sb`: Stolen bases
- Target columns (e.g., `mlb_avg`, `mlb_war`)

### Unstructured Data (scouting_reports.csv)

Required columns:

- `player_id`: Unique identifier
- `report_date`: Date of report
- `scout_name`: Scout name
- `report_text`: Full scouting report text

## Configuration

Edit `config/config.yaml` to customize:

- Data paths
- NLP settings (models, processing options)
- Feature engineering options
- Model hyperparameters
- Training parameters
- Evaluation metrics

## Advanced Features

### Ensemble Predictions

```python
# Create ensemble from multiple models
from src.models import EnsembleModel

ensemble = EnsembleModel(
    models=[rf_model, xgb_model, lgb_model],
    weights=[0.3, 0.4, 0.3]
)

predictions, uncertainty = ensemble.predict_with_uncertainty(X_test)
```

### Feature Importance

```python
# Get feature importance
importance = model.get_feature_importance()
print(importance.head(20))

# Plot feature importance
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.plot_feature_importance(model, 'XGBoost', 'mlb_war')
```

### Custom NLP Features

The NLP pipeline automatically extracts:

- Tool grades (20-80 scale)
- Sentiment analysis
- Keyword mentions (strengths, concerns)
- Skill detection (power, speed, contact, etc.)

## Troubleshooting

### Common Issues

1. **Missing spaCy model**

   ```bash
   python -m spacy download en_core_web_lg
   ```

2. **Import errors**

   - Make sure you're in the project root directory
   - Check that all dependencies are installed: `pip install -r requirements.txt`

3. **Data not found**

   - Generate sample data: `python scripts/generate_sample_data.py`
   - Check data paths in `config/config.yaml`

4. **GPU/CUDA issues (for PyTorch)**
   - Deep learning models will automatically use CPU if CUDA is not available
   - To force CPU: Set `CUDA_VISIBLE_DEVICES=""` environment variable

## Next Steps

1. Replace sample data with your actual prospect data
2. Adjust feature engineering based on your data characteristics
3. Tune model hyperparameters using grid search
4. Add custom features specific to your use case
5. Implement cross-validation for more robust evaluation
6. Deploy models as a REST API or web application

## Resources

- Baseball Prospectus: https://www.baseballprospectus.com/
- FanGraphs: https://www.fangraphs.com/
- MLB Statcast: https://baseballsavant.mlb.com/
