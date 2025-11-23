# ScoutIQ - Baseball Prospect Projection System

An advanced NLP and AI solution for analyzing baseball prospect data and projecting major league performance.

## Overview

ScoutIQ combines structured statistical data with unstructured scouting reports to generate comprehensive player projections using:

- Natural Language Processing (spaCy, transformers)
- Machine Learning (scikit-learn, XGBoost, LightGBM)
- Deep Learning (PyTorch)
- Feature engineering from both structured and unstructured data

## Architecture

```
scoutiq/
├── data/                          # Data storage
│   ├── raw/                       # Raw data sources
│   ├── processed/                 # Cleaned and processed data
│   └── models/                    # Trained model artifacts
├── src/
│   ├── data_ingestion/           # Data loading and parsing
│   ├── nlp/                      # NLP processing pipeline
│   ├── features/                 # Feature engineering
│   ├── models/                   # ML models and training
│   ├── evaluation/               # Model evaluation
│   └── utils/                    # Utilities
├── notebooks/                     # Jupyter notebooks for analysis
├── tests/                        # Unit tests
└── config/                       # Configuration files
```

## Features

### Data Ingestion

- **Structured Data**: Player statistics (batting, pitching, fielding)
- **Unstructured Data**: Scouting reports, player descriptions, injury reports

### NLP Pipeline

- Text preprocessing and cleaning
- Named Entity Recognition (NER) for player attributes
- Sentiment analysis for scouting opinions
- Tool grade extraction (20-80 scale)
- Skill and attribute extraction
- Injury and concern identification

### Feature Engineering

- Statistical aggregations and trends
- Advanced metrics (wOBA, FIP, WAR projections)
- Age-adjusted performance curves
- NLP-derived features (tool grades, scout sentiment)
- Composite features combining stats and scouting

### Machine Learning Models

- Ensemble methods (Random Forest, XGBoost, LightGBM)
- Deep learning (LSTM for time series, transformer models)
- Multi-task learning (multiple projection targets)
- Uncertainty quantification

### Projection Targets

- Major league batting stats (AVG, OBP, SLG, HR, etc.)
- Major league pitching stats (ERA, WHIP, K/9, BB/9, etc.)
- WAR projections
- ETA (Estimated Time of Arrival) to majors
- Success probability classifications

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg
```

## Quick Start

```python
from src.pipeline import ProspectProjectionPipeline

# Initialize pipeline
pipeline = ProspectProjectionPipeline()

# Load data
pipeline.load_data(
    stats_file='data/raw/prospect_stats.csv',
    reports_file='data/raw/scouting_reports.csv'
)

# Process and train
pipeline.process()
pipeline.train()

# Generate projections
projections = pipeline.predict(player_id='player_001')
print(projections)
```

## Data Format

### Structured Data (CSV/JSON)

```csv
player_id,name,age,level,pa,avg,obp,slg,hr,sb,bb_rate,k_rate
P001,John Doe,21,AA,450,.285,.365,.475,18,12,10.2,22.8
```

### Unstructured Data (Text)

```json
{
  "player_id": "P001",
  "report_date": "2025-06-15",
  "scout_name": "Jane Smith",
  "report": "Plus bat speed with raw power potential. Shows good plate discipline..."
}
```

## Configuration

Edit `config/config.yaml` to customize:

- Model hyperparameters
- Feature selection
- Training parameters
- Data paths

## License

MIT License
