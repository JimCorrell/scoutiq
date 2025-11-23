# Lahman Baseball Database Integration

## Overview

The Lahman Baseball Database (1871-2024) is now integrated into ScoutIQ, providing comprehensive historical baseball statistics for building projection models.

## Data Structure

```
data/
└── lahman/
    └── lahman_1871-2024u_csv/
        ├── Batting.csv         (126,908 records)
        ├── Pitching.csv        (56,616 records)
        ├── People.csv          (24,024 players)
        ├── Fielding.csv
        ├── Teams.csv
        └── [20+ additional files]
```

## Quick Start

### 1. Load Historical Data

```python
from src.data_ingestion import LahmanDataLoader

loader = LahmanDataLoader()

# Load batting statistics (2000+)
batting = loader.load_batting(min_year=2000)

# Get comprehensive player-season records
player_seasons = loader.create_player_seasons(
    min_year=2000,
    min_plate_appearances=200
)
```

### 2. Prepare Projection Data

```python
# Create features from 3-year history to predict next season
features, targets = loader.prepare_projection_data(
    current_year=2023,
    lookback_years=3,
    target_year_offset=1
)
```

### 3. Train Projection Models

```python
from src.models import RandomForestModel

# Train model to project batting average
model = RandomForestModel()
model.train(features, targets['AVG'])

# Make projections
predictions = model.predict(new_player_data)
```

## Example Scripts

### Run Complete Example

```bash
python examples/lahman_projections.py
```

This will:

- Load 24 years of MLB data (2000-2024)
- Create player-season records
- Build projection models for AVG, OBP, SLG, HR
- Evaluate model performance
- Save trained models

### Explore in Notebook

```bash
jupyter notebook notebooks/lahman_exploration.md
```

Interactive exploration of:

- Offensive trends over time
- Career trajectories
- Age curves
- Feature engineering
- Model training and evaluation

## Available Data

### Core Datasets

- **Batting**: 126K+ records of player batting statistics
- **Pitching**: 56K+ records of pitcher statistics
- **People**: 24K+ player biographical data
- **Fielding**: Defensive statistics by position
- **Teams**: Team-level statistics and standings

### Key Metrics Available

**Batting:**

- Traditional: AVG, OBP, SLG, OPS
- Counting stats: G, AB, H, 2B, 3B, HR, RBI, SB, BB, SO
- Advanced: Can calculate wOBA, ISO, BABIP, etc.

**Biographical:**

- Birth date/place, height, weight
- Bats/throws
- Debut and final game dates

## Methodology

### Player-Season Creation

The `create_player_seasons()` method:

1. Loads batting, fielding, and biographical data
2. Calculates derived metrics (PA, rate stats)
3. Merges player information
4. Filters by minimum PA threshold
5. Assigns primary position
6. Returns comprehensive player-season records

### Projection Data Preparation

The `prepare_projection_data()` method:

1. Aggregates N years of historical data per player
2. Calculates mean and std for key metrics
3. Links to target year performance
4. Creates train/test splits
5. Returns features and targets ready for modeling

## Use Cases

### 1. Build Baseline Projections

Use Lahman data to create baseline projections for established MLB players:

```python
# Get last 3 years of player history
features = loader.create_player_seasons(min_year=2021)

# Train on this data
model.train(features[feature_cols], features['target_metric'])
```

### 2. Career Trajectory Analysis

Analyze typical career paths by position:

```python
trajectories = loader.create_career_trajectories(
    min_seasons=5,
    min_year=2000
)

# Analyze peak age by position
peak_analysis = trajectories['trajectories'].groupby('POS').apply(
    lambda x: x.groupby('age')['OPS'].mean().idxmax()
)
```

### 3. Combine with Prospect Data

Use Lahman as training data for prospect projections:

```python
# Train on established players
lahman_features, lahman_targets = loader.prepare_projection_data()
model.train(lahman_features, lahman_targets)

# Apply to prospects with similar features
prospect_projections = model.predict(prospect_features)
```

## Data Quality Notes

- Data is complete through 2024 season
- Modern era (2000+) recommended for projections due to game evolution
- Missing values are common in older records
- Minimum PA thresholds recommended (100-300 depending on use case)
- Pitcher batting statistics available but generally excluded from hitter models

## Integration with ScoutIQ Pipeline

The Lahman loader integrates seamlessly with existing ScoutIQ components:

```python
from src.pipeline import ProspectProjectionPipeline
from src.data_ingestion import LahmanDataLoader

# Load historical data
lahman = LahmanDataLoader()
player_seasons = lahman.create_player_seasons(min_year=2015)

# Use with existing pipeline
pipeline = ProspectProjectionPipeline(config_path="config/config.yaml")

# Train models on Lahman data
pipeline.train(player_seasons[feature_cols], player_seasons[target_cols])

# Apply to new prospects
projections = pipeline.predict(prospect_data)
```

## References

- **Lahman Database**: [seanlahman.com](http://www.seanlahman.com/baseball-archive/statistics/)
- **Data Dictionary**: See `data/lahman/readme.txt` for column definitions
- **Updates**: Database typically updated after each season

## Next Steps

1. **Feature Engineering**: Add park factors, league adjustments, aging curves
2. **Time Series**: Implement rolling averages and trend analysis
3. **Ensemble Models**: Combine multiple projection systems
4. **Uncertainty**: Add confidence intervals to projections
5. **Real-time Updates**: Integrate with current season data feeds
