# Getting Started with ScoutIQ

## 🎯 What You Have Now

ScoutIQ is a **baseball prospect projection system** that combines:

- Historical MLB statistics (Lahman Database 1871-2024)
- NLP processing for scouting reports
- Machine learning models (Random Forest, XGBoost, LightGBM, Neural Networks)
- Complete feature engineering pipeline

## 📁 Project Structure

```
scoutiq/
├── src/                          # Core source code
│   ├── data_ingestion/          # Load and integrate data
│   │   ├── loaders.py           # Generic CSV/JSON loaders
│   │   └── lahman_loader.py     # Historical MLB data loader ⭐
│   ├── nlp/                     # Natural language processing
│   │   └── processor.py         # Extract features from scouting reports
│   ├── features/                # Feature engineering
│   │   └── engineering.py       # Create 60+ statistical features
│   ├── models/                  # Machine learning models
│   │   └── models.py            # RF, XGBoost, LightGBM, Neural Nets
│   ├── evaluation/              # Model evaluation
│   │   └── evaluator.py         # Metrics, plots, reports
│   └── utils/                   # Utilities
│       ├── config.py            # Configuration management
│       └── logger.py            # Logging setup
│
├── data/                        # Data storage
│   ├── lahman/                  # 📊 Historical MLB stats (126K+ records)
│   ├── raw/                     # Your input data goes here
│   ├── processed/               # Processed data
│   └── models/                  # Saved model files
│
├── examples/                    # Example scripts
│   ├── lahman_projections.py   # 🚀 START HERE - Full working example
│   └── comprehensive_example.py # Original demo (uses synthetic data)
│
├── notebooks/                   # Jupyter notebooks
│   └── lahman_exploration.md   # Interactive data exploration
│
├── scripts/                     # Utility scripts
│   ├── generate_sample_data.py # Generate synthetic test data
│   └── run_example.py          # Quick demo
│
├── config/
│   └── config.yaml             # System configuration
│
└── Documentation
    ├── README.md               # Project overview
    ├── SETUP.md                # Detailed setup guide
    ├── LAHMAN_INTEGRATION.md   # Historical data guide ⭐
    └── GETTING_STARTED.md      # This file!
```

## 🚀 Quick Start (5 minutes)

### 1. Run Your First Projection Model

The easiest way to see everything in action:

```bash
# Make sure you're in the project directory
cd /Users/jimcorrell/Development/neho/scoutiq

# Activate virtual environment
source .venv/bin/activate

# Run the Lahman projections example
python examples/lahman_projections.py
```

This will:

- Load 24 years of MLB batting statistics (2000-2024)
- Create 3-year historical features for each player
- Train Random Forest models to predict AVG, OBP, SLG, HR
- Show you feature importance and sample predictions
- Save trained models to `data/models/lahman_projections/`

**Expected output:**

```
1. Loading Lahman Baseball Database...
   ✓ Loaded 80,000+ player-seasons from 2000-2024

2. Preparing projection training data...
   ✓ Created 15,000+ training samples

3. Training projection models...
   Training AVG projection model...
      Test MAE:  0.0180
      Test R²:   0.6234
```

### 2. Explore the Data Interactively

```bash
# Install Jupyter if you haven't
pip install jupyter matplotlib seaborn

# Start Jupyter
jupyter notebook notebooks/lahman_exploration.md
```

This notebook shows:

- Offensive trends over time (home runs, batting average)
- Career trajectory analysis
- Age curves (when players peak)
- Feature correlation analysis
- Quick model training and evaluation

### 3. Understand Your Data

```python
from src.data_ingestion import LahmanDataLoader

# Initialize loader
loader = LahmanDataLoader()

# Load recent seasons
player_seasons = loader.create_player_seasons(
    min_year=2020,
    min_plate_appearances=200
)

print(f"Loaded {len(player_seasons)} player-seasons")
print(f"Available stats: {list(player_seasons.columns)}")

# See a sample player
sample = player_seasons.iloc[0]
print(f"{sample['nameFirst']} {sample['nameLast']}")
print(f"  {sample['yearID']}: {sample['AVG']:.3f} AVG, {sample['HR']} HR")
```

## 📊 What's in the Lahman Database?

Your `data/lahman/` folder contains comprehensive MLB history:

- **126,908 batting records** - Every player-season from 1871-2024
- **56,616 pitching records** - Complete pitching statistics
- **24,024 player records** - Biographical data (age, position, etc.)
- **Plus**: Fielding, teams, awards, salaries, and more

Key metrics available:

- Traditional: AVG, OBP, SLG, OPS
- Counting: G, AB, H, 2B, 3B, HR, RBI, SB, BB, SO
- Advanced: Can calculate wOBA, ISO, BABIP, wRC+, etc.

## 🎓 Key Concepts

### 1. Player-Season Records

A "player-season" is one player's stats for one year:

```python
# Aaron Judge, 2022:
# - 157 games, 696 PA, .311 AVG, 62 HR, 1.111 OPS
```

### 2. Projection Method

The system uses historical patterns to predict future performance:

1. **Input**: Player's last 3 years of stats
2. **Model**: Random Forest learns from 20+ years of data
3. **Output**: Predicted stats for next season + uncertainty

### 3. Features

The system creates 60+ features from raw data:

- Rate stats (AVG, OBP, SLG)
- Trends (improving/declining)
- Consistency (standard deviation)
- Age adjustments
- Position factors

## 🔧 Common Tasks

### Train a Model for a Specific Metric

```python
from src.data_ingestion import LahmanDataLoader
from src.models import RandomForestModel
from sklearn.model_selection import train_test_split

# Get data
loader = LahmanDataLoader()
features, targets = loader.prepare_projection_data(
    current_year=2023,
    lookback_years=3
)

# Prepare features
X = features.drop('playerID', axis=1).fillna(0)
y = targets['HR']  # Predict home runs

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestModel()
model.train(X_train, y_train, task='regression')

# Evaluate
predictions = model.predict(X_test)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predictions)
print(f"Home Run Prediction MAE: {mae:.2f}")
```

### Add Your Own Data

Place CSV files in `data/raw/`:

```
data/raw/
├── my_prospects.csv       # Your prospect data
└── scouting_reports.csv   # Text scouting reports
```

Then load and merge:

```python
from src.data_ingestion import StructuredDataLoader

loader = StructuredDataLoader("data/raw")
my_data = loader.load_csv("my_prospects.csv")

# Merge with Lahman historical data
combined = merge_prospect_with_historical(my_data, player_seasons)
```

### Customize Model Configuration

Edit `config/config.yaml`:

```yaml
models:
  random_forest:
    n_estimators: 200 # More trees = better accuracy
    max_depth: 15 # Deeper = more complex patterns
    min_samples_leaf: 5 # Smaller = more detail
```

## 🎯 Next Steps

### Beginner Track

1. ✅ Run `examples/lahman_projections.py`
2. 📊 Explore data in the Jupyter notebook
3. 🎨 Try predicting different metrics (SB, RBI, OPS)
4. 📈 Visualize predictions vs actuals

### Intermediate Track

1. 🔧 Add new features (career trends, park factors)
2. 🤖 Try different models (XGBoost, LightGBM)
3. 📊 Build ensemble predictions
4. 🎯 Add confidence intervals

### Advanced Track

1. 🔍 Integrate prospect scouting reports (NLP)
2. 🎓 Implement aging curves
3. ⚾ Add park factor adjustments
4. 🌐 Build a web API for predictions

## 🆘 Troubleshooting

### "ModuleNotFoundError"

```bash
# Install dependencies
source .venv/bin/activate
pip install -r requirements.txt
```

### "FileNotFoundError: Lahman directory"

The Lahman data should be in:

```
data/lahman/lahman_1871-2024u_csv/
```

Check that this directory exists with CSV files.

### "No module named 'src'"

```bash
# Make sure you're in the project root
cd /Users/jimcorrell/Development/neho/scoutiq

# Run from project root
python examples/lahman_projections.py
```

## 📚 Learn More

- **LAHMAN_INTEGRATION.md** - Deep dive into historical data
- **SETUP.md** - Detailed installation and configuration
- **README.md** - Architecture and system design
- **Code examples** - Look in `examples/` and `notebooks/`

## 💡 Tips

1. **Start small**: Use recent years (2020+) for faster testing
2. **Check data**: Always inspect a few samples before training
3. **Validate**: Split data chronologically (train on 2000-2022, test on 2023)
4. **Iterate**: Start with simple models, add complexity gradually
5. **Document**: Keep notes on what works and what doesn't

## 🎉 You're Ready!

You now have a complete, working baseball projection system with 24 years of historical data. Start with the examples, experiment with the data, and build something awesome!

Questions? Check the documentation or examine the example code.

**Happy projecting! ⚾**
