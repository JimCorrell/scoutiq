# Lahman Baseball Database - Data Exploration

This notebook explores the Lahman Baseball Database and builds projection models.

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
import sys
sys.path.append('..')

from src.data_ingestion import LahmanDataLoader
from src.utils.logger import setup_logger

logger = setup_logger('lahman_notebook')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

%matplotlib inline
sns.set_style('whitegrid')
```

## 1. Load Lahman Data

```python
# Initialize loader
loader = LahmanDataLoader()

# Load key datasets
batting = loader.load_batting(min_year=2000)
pitching = loader.load_pitching(min_year=2000)
people = loader.load_people()

print(f"Batting records: {len(batting):,}")
print(f"Pitching records: {len(pitching):,}")
print(f"Players: {len(people):,}")
```

## 2. Explore Batting Statistics

```python
# Look at recent seasons
recent_batting = batting[batting['yearID'] >= 2020].copy()

# Calculate PA and rate stats
recent_batting['PA'] = (
    recent_batting['AB'] +
    recent_batting['BB'].fillna(0) +
    recent_batting['HBP'].fillna(0) +
    recent_batting['SH'].fillna(0) +
    recent_batting['SF'].fillna(0)
)

recent_batting['AVG'] = recent_batting['H'] / recent_batting['AB']
recent_batting['OBP'] = (
    (recent_batting['H'] + recent_batting['BB'] + recent_batting['HBP']) /
    recent_batting['PA']
)

# Filter to qualified batters (>= 300 PA)
qualified = recent_batting[recent_batting['PA'] >= 300].copy()

print(f"Qualified batters (2020-2024): {len(qualified)}")
qualified.head()
```

## 3. Visualize Offensive Trends

```python
# Home runs per season
hr_by_year = batting.groupby('yearID')['HR'].sum()

plt.figure(figsize=(12, 6))
plt.plot(hr_by_year.index, hr_by_year.values, linewidth=2)
plt.title('Total Home Runs per Season (2000-2024)', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Total Home Runs')
plt.grid(True, alpha=0.3)
plt.show()
```

```python
# Batting average distribution over time
plt.figure(figsize=(14, 6))

for year in [2000, 2010, 2020, 2023]:
    year_data = batting[batting['yearID'] == year].copy()
    year_data = year_data[year_data['AB'] >= 300]
    year_data['AVG'] = year_data['H'] / year_data['AB']

    plt.hist(year_data['AVG'], bins=30, alpha=0.5, label=str(year))

plt.xlabel('Batting Average')
plt.ylabel('Frequency')
plt.title('Batting Average Distribution by Era')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 4. Create Player Career Trajectories

```python
# Get career data
careers = loader.create_career_trajectories(min_seasons=5, min_year=2000)
trajectories = careers['trajectories']

print(f"Players with 5+ seasons: {len(careers['qualified_players'])}")

# Look at a sample player's career
sample_player = trajectories['playerID'].value_counts().index[0]
player_career = trajectories[trajectories['playerID'] == sample_player].sort_values('yearID')

plt.figure(figsize=(12, 6))
plt.plot(player_career['yearID'], player_career['OPS'], marker='o', linewidth=2, markersize=8)
plt.title(f"Career OPS Trajectory: {sample_player}", fontsize=14)
plt.xlabel('Year')
plt.ylabel('OPS')
plt.grid(True, alpha=0.3)
plt.show()
```

## 5. Age Curves Analysis

```python
# Calculate average performance by age
player_seasons = loader.create_player_seasons(min_year=2000, min_plate_appearances=300)

age_performance = player_seasons.groupby('age').agg({
    'AVG': 'mean',
    'OBP': 'mean',
    'SLG': 'mean',
    'OPS': 'mean',
    'HR': 'mean'
}).reset_index()

# Filter to reasonable age range
age_performance = age_performance[(age_performance['age'] >= 20) & (age_performance['age'] <= 40)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['AVG', 'OBP', 'SLG', 'HR']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    ax.plot(age_performance['age'], age_performance[metric], marker='o', linewidth=2)
    ax.set_xlabel('Age')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by Age (2000-2024)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 6. Prepare Projection Data

```python
# Create training data for projections
features, targets = loader.prepare_projection_data(
    current_year=2023,
    lookback_years=3,
    target_year_offset=1
)

print(f"Training samples: {len(features)}")
print(f"Feature columns: {len(features.columns)}")
print(f"Target columns: {list(targets.columns)}")

# Show feature correlations with targets
feature_cols = [col for col in features.columns if col not in ['playerID']]
X = features[feature_cols].fillna(0)

# Correlation with AVG
if 'AVG' in targets.columns:
    correlations = pd.DataFrame({
        'feature': X.columns,
        'correlation': [X[col].corr(targets['AVG']) for col in X.columns]
    }).sort_values('correlation', ascending=False, key=abs)

    print("\nTop 10 features correlated with AVG:")
    print(correlations.head(10))
```

## 7. Quick Model Training

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Prepare data
X = features[feature_cols].fillna(0)
y = targets['AVG'].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(f"Train MAE: {mean_absolute_error(y_train, train_pred):.4f}")
print(f"Test MAE:  {mean_absolute_error(y_test, test_pred):.4f}")
print(f"Test R²:   {r2_score(y_test, test_pred):.4f}")

# Plot predictions vs actuals
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual AVG')
plt.ylabel('Predicted AVG')
plt.title('Batting Average Projections')
plt.grid(True, alpha=0.3)
plt.show()
```

## 8. Feature Importance

```python
# Get feature importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(importance_df.head(15)['feature'], importance_df.head(15)['importance'])
plt.xlabel('Importance')
plt.title('Top 15 Most Important Features for AVG Projection')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

## Next Steps

1. Add more sophisticated features (rolling averages, trends, etc.)
2. Incorporate park factors and league adjustments
3. Build models for multiple target metrics
4. Create ensemble models
5. Add uncertainty estimates to projections
6. Integrate with scouting data for prospects
