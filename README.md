# Ultra-Marathon Pace Prediction Project

This project implements a machine learning pipeline to predict ultramarathon finishing times using historical race data. The code is organized following the Cookiecutter Data Science template structure for better maintainability and reproducibility.

## Project Structure

```
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned and feature-engineered data
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── src/                        # Source code modules
│   ├── data/                   # Data processing modules
│   │   ├── load.py            # Data loading functions
│   │   ├── clean.py           # Data cleaning functions
│   │   ├── features.py        # Feature engineering functions
│   │   └── split.py           # Train/test splitting functions
│   ├── models/                 # Machine learning modules
│   │   ├── prepare.py         # Feature preparation for modeling
│   │   └── train.py           # Model training functions
│   ├── visualization/          # Visualization modules
│   │   └── eda.py             # EDA plotting functions
│   └── evaluation/             # Model evaluation modules
│       └── metrics.py         # Custom evaluation metrics
├── tests/                      # Unit tests
│   ├── test_data.py           # Tests for data modules
│   └── test_models.py         # Tests for model modules
├── reports/                    # Generated reports and figures
└── src/pipeline.py            # Main pipeline orchestration
```

## Key Features

### Data Processing
- **Event Name Cleaning**: Standardizes race names by removing years, country codes, and noise
- **Distance Extraction**: Automatically extracts numeric distances from text descriptions
- **Time Conversion**: Converts race times to consistent formats
- **Missing Value Handling**: Removes irrelevant columns and rows with critical missing data

### Feature Engineering
- **Cumulative Statistics**: Tracks athlete progression over time (races completed, average pace, best pace)
- **Distance Analysis**: Tracks shortest/longest distances, recent average distance
- **Experience Metrics**: Counts Western States finishes and other experience indicators
- **Athlete Demographics**: Calculates age at time of race

### Machine Learning
- **Target Encoding**: Uses smoothed target encoding for race difficulty
- **LightGBM**: Gradient boosting model optimized for large datasets
- **Time-based Splitting**: Prevents data leakage by ensuring no future information in training
- **Comprehensive Evaluation**: Multiple metrics including pace-specific accuracy measures

## Usage

### Running the Pipeline

```python
from src.pipeline import run_pipeline

# Run the complete pipeline
model, X_train, X_test, y_train, y_test, y_pred = run_pipeline('path/to/data.csv')
```

### Individual Components

```python
# Load data
from src.data.load import load_raw_data
df = load_raw_data('data.csv')

# Clean data
from src.data.clean import clean_data
df_clean = clean_data(df)

# Engineer features
from src.data.features import engineer_features
df_features = engineer_features(df_clean)

# Split data
from src.data.split import split_train_test
df_train, df_test, feature_cols = split_train_test(df_features)

# Prepare for modeling
from src.models.prepare import prepare_model_data
X_train, X_test, y_train, y_test = prepare_model_data(df_train, df_test, feature_cols)

# Train model
from src.models.train import train_evaluate_lgbm
model, y_pred = train_evaluate_lgbm(X_train, y_train, X_test, y_test)
```

### Evaluation

```python
from src.evaluation.metrics import calculate_pace_metrics, print_pace_metrics

# Calculate metrics
metrics = calculate_pace_metrics(y_test, y_pred)

# Print formatted results
print_pace_metrics(y_test, y_pred, "LightGBM Model")
```

## Key Metrics

The model is evaluated using several pace-specific metrics:

- **MAE**: Mean Absolute Error (minutes per kilometer)
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **Accuracy**: Percentage of predictions within 30 seconds, 1 minute, and 2 minutes per km

## Data Requirements

The pipeline expects a CSV file with the following columns:
- `Year of event`: Year the race took place
- `Event name`: Name of the race
- `Event distance/length`: Distance of the race (with units)
- `Athlete performance`: Race finishing time
- `Athlete gender`: Gender of the athlete
- `Athlete year of birth`: Birth year of the athlete
- `Athlete age category`: Age category of the athlete
- `Athlete ID`: Unique identifier for the athlete

## Dependencies

- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- seaborn

## Testing

Run the tests with:

```bash
python -m pytest tests/
```

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for:
1. **Data Exploration**: Understanding the dataset structure and distributions
2. **Data Cleaning**: Demonstrating the cleaning process with examples
3. **Feature Engineering**: Showing how features are created and their distributions
4. **Modeling**: Complete modeling workflow with hyperparameter tuning

## Reproducibility

The pipeline is designed to be reproducible:
- All random seeds are set for consistent results
- Data processing is deterministic
- Model parameters are documented and configurable
- Results can be saved and loaded for comparison

## Performance

The model is optimized for:
- Large datasets (millions of records)
- Time-series data with athlete progression
- Mixed data types (numerical, categorical, temporal)
- Real-world constraints (no future data leakage)
