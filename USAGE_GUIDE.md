# Ultra-Marathon Pace Prediction - Usage Guide

This guide explains how to use the complete ultramarathon pace prediction pipeline with your data.

## 🚀 Quick Start

### 1. Install Requirements

First, install all required dependencies:

```bash
python install_requirements.py
```

This script will:
- Check your Python environment
- Install all required packages from `requirements.txt`
- Verify the installations

### 2. Run the Pipeline

Execute the complete pipeline with your data:

```bash
python run_pipeline.py
```

The script will:
- Automatically find your CSV file in `data/raw/`
- Run the complete pipeline (load → clean → feature engineering → train → evaluate)
- Display results and save outputs to `results/` directory

## 📁 Project Structure

```
├── data/
│   ├── raw/                    # Your CSV file goes here
│   └── processed/              # Cleaned data (created by pipeline)
├── src/                        # Source code modules
├── notebooks/                  # Jupyter notebooks for exploration
├── tests/                      # Unit tests
├── results/                    # Pipeline outputs (created by run_pipeline.py)
├── reports/                    # Generated plots and figures
├── run_pipeline.py            # Main pipeline runner
└── install_requirements.py    # Dependency installer
```

## 📊 Data Requirements

Your CSV file should contain these columns:

- `Year of event`: Year the race took place (e.g., 2022)
- `Event name`: Name of the race (e.g., "Western States 2022")
- `Event distance/length`: Distance with units (e.g., "100km", "50 miles")
- `Athlete performance`: Race finishing time (e.g., "4:30:15", "2d 6:45:20")
- `Athlete gender`: Gender of the athlete (e.g., "M", "F")
- `Athlete year of birth`: Birth year (e.g., 1985)
- `Athlete age category`: Age category (e.g., "M35", "F25")
- `Athlete ID`: Unique identifier for the athlete

## 🎯 Pipeline Output

The pipeline is now modular with separated CV and final training flows. Choose the appropriate pipeline for your needs:

### Data Processing Only
```bash
python -c "from src.pipeline_data import run_polars_pipeline; run_polars_pipeline('data/raw/TWO_CENTURIES_OF_UM_RACES.csv')"
```

### Cross-Validation Only (for model refinement)
```bash
python -c "from src.pipeline_full import run_cv_only_pipeline; run_cv_only_pipeline('data/raw/TWO_CENTURIES_OF_UM_RACES.csv')"
```
This performs CV on training data only, saves CV metrics, and allows iterative model refinement without training a final model.

### Final Training Only (production model)
```bash
python -c "from src.pipeline_full import run_final_training_only_pipeline; run_final_training_only_pipeline('data/raw/TWO_CENTURIES_OF_UM_RACES.csv')"
```
This trains a final model on complete training data and evaluates on test set. Run this after satisfactory CV performance.

### Complete Pipeline (CV + Final Training)
```bash
python -c "from src.pipeline_full import run_full_ml_pipeline; run_full_ml_pipeline('data/raw/TWO_CENTURIES_OF_UM_RACES.csv')"
```

### ML Training Only (on pre-processed data)
```bash
python -c "from src.pipeline_ml import run_ml_pipeline; run_ml_pipeline(processed_data)"
```

## 📁 New Modular Structure

```
src/
├── pipeline_data.py               # Data processing pipeline
├── pipeline_ml.py                 # Combined ML training + CV pipeline
├── pipeline_cv.py                 # CV-only evaluation pipeline
├── pipeline_train_final.py        # Final model training pipeline
├── pipeline_ml_from_features.py   # ML training from saved features
└── pipeline_full.py               # Complete end-to-end pipelines (all modes)
```

### Console Output
- Data loading and cleaning statistics
- Feature engineering results
- **Cross-validation results on training set** (for model refinement)
- Model training progress
- Performance metrics (MAE, RMSE, R², accuracy percentages)
- Feature importance rankings

### Saved Files (in `training_results/` directory)
- `lightgbm_model.pkl`: Trained LightGBM model
- `test_predictions.csv`: Actual vs predicted paces with errors
- `feature_importance.csv`: Feature importance rankings
- `cv_metrics.txt`: Cross-validation performance metrics on training set

## 🔧 Individual Component Usage

You can also run individual components for learning or debugging:

### Load and Explore Data
```python
from src.data.load import load_raw_data
df = load_raw_data("data/raw/your_data.csv")
print(f"Loaded {len(df)} records")
```

### Clean Data
```python
from src.data.clean import clean_data
df_clean = clean_data(df)
print(f"Cleaned data: {df_clean.shape}")
```

### Engineer Features
```python
from src.features.build_features import engineer_features
df_features = engineer_features(df_clean)
print(f"Features created: {df_features.shape}")
```

### Run Modular Pipelines
```python
# Data processing only
from src.pipeline_data import run_polars_pipeline_with_collection
processed_data = run_polars_pipeline_with_collection('data/raw/TWO_CENTURIES_OF_UM_RACES.csv')

# ML training only
from src.pipeline_ml import run_ml_pipeline
results = run_ml_pipeline(processed_data)

# ML training from saved features (fast iteration)
from src.pipeline_ml_from_features import run_ml_from_features
results = run_ml_from_features()  # Loads from data/processed/final_features.parquet

# Complete pipeline
from src.pipeline_full import run_full_ml_pipeline
full_results = run_full_ml_pipeline('data/raw/TWO_CENTURIES_OF_UM_RACES.csv')
```

### Fast ML Iteration Workflow

For rapid ML experimentation without re-processing data:

```bash
# 1. Run data processing once (expensive)
python -c "from src.pipeline_data import run_polars_pipeline; run_polars_pipeline('data/raw/TWO_CENTURIES_OF_UM_RACES.csv')"

# 2. Iterate on ML code quickly (fast)
python -c "from src.pipeline_ml_from_features import run_ml_from_features; run_ml_from_features()"

# 3. Inspect your features file
python -c "from src.pipeline_ml_from_features import inspect_features_file; inspect_features_file()"

# 4. Check cross-validation metrics for model refinement
cat training_results/cv_metrics.txt
```

**Note**: The pipeline now includes 5-fold cross-validation on the training set to help refine your model without overfitting on the test set (Western States 2022).

## 📓 Jupyter Notebooks

For interactive exploration, use the notebooks in `notebooks/`:

- `example_usage.ipynb`: Complete walkthrough with explanations
- Individual notebooks for each pipeline stage (01-04)

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python -m pytest tests/
```

## 📈 Understanding Results

### Performance Metrics
- **MAE**: Mean Absolute Error (minutes per kilometer) - lower is better
- **RMSE**: Root Mean Squared Error - lower is better
- **R²**: Coefficient of determination (0-1, higher is better)
- **Accuracy**: Percentage of predictions within time thresholds
- **CV Metrics**: Cross-validation MAE/RMSE on training set (for model refinement) - compare to final test metrics to assess overfitting

### Feature Importance
The pipeline identifies which features most influence predictions:
1. Cumulative athlete statistics (experience, progression)
2. Race characteristics (distance, difficulty)
3. Athlete demographics (age, gender)

## 🐛 Troubleshooting

### Common Issues

**"No CSV files found in data/raw/"**
- Ensure your CSV file is in the `data/raw/` directory
- File must have `.csv` extension

**Import errors**
- Run `python install_requirements.py` to install dependencies
- Check Python version (requires 3.8+)

**Memory errors with large datasets**
- The pipeline is optimized for large datasets but may need more RAM
- Consider running on a machine with more memory

**Poor model performance**
- Check data quality and completeness
- Ensure target variable (pace) is properly calculated
- Verify feature engineering steps completed successfully

### Getting Help

1. Check the console output for specific error messages
2. Verify your data format matches the requirements
3. Run individual components to isolate issues
4. Check the test suite for validation

## 🎓 Learning Path

For students learning the pipeline:

1. **Start with notebooks**: `notebooks/example_usage.ipynb`
2. **Understand each module**: Read the source code in `src/`
3. **Run individual components**: Practice with small datasets
4. **Modify and experiment**: Try different features or models
5. **Test your changes**: Use the test suite to verify modifications

## 🚀 Advanced Usage

### Custom Model Parameters
```python
from src.models.train import train_evaluate_lgbm

# Custom LightGBM parameters
custom_params = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "num_leaves": 128
}

model, y_pred = train_evaluate_lgbm(X_train, y_train, X_test, y_test, params=custom_params)
```

### Custom Data Processing
```python
from src.data.clean import clean_data, clean_event_name

# Custom event name cleaning
def custom_clean_event_name(name):
    # Your custom logic here
    return clean_event_name(name)

# Apply custom cleaning
df['Event_name_custom'] = df['Event name'].apply(custom_clean_event_name)
```

## 📞 Support

For questions or issues:
1. Check this guide first
2. Review the source code comments
3. Run the test suite for validation
4. Check the example notebooks for usage patterns

The pipeline is designed to be educational, so feel free to explore and modify the code!
