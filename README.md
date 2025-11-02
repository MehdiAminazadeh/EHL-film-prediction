# EHL Film Prediction – Streamlit App

This repository contains a Streamlit-based machine learning application for predicting **Average Film Thickness (nm)** from experimental EHL data.  
It supports classical ML, deep learning (TabNet, NODE, FT-Transformer), and statistical visualization — all inside an interactive web interface.

---

## Author
**Mehdi Aminazadeh**  
Master of Computer Science, RPTU  
m.aminazadeh@edu.rptu.de

---

## Project Overview

The project enables researchers to:
1. Upload and preprocess raw `.txt` or `.zip` experiment data.
2. Clean, merge, and impute missing data with **KNN**.
3. Explore data through interactive statistics and plots.
4. Train and compare models (classical ML & DL) to predict film thickness.
5. Export cleaned data and model predictions.

---

## Project Structure

```
EHL Film Prediction/
│
├── .streamlit/                  # Streamlit configuration files
├── assets/                      # Static resources (logos, icons)
├── configs/                     # Configuration templates
├── data/                        # Local data storage (temporary & cleaned data)
│
├── machine-learning-for-film-thickness-pred/   # Previous experiments / archives
│
├── models/                      # ML model adapters and registry
│   ├── __init__.py
│   ├── adapters.py
│   ├── base.py
│   ├── classical.py
│   ├── dnn.py
│   ├── dnn_adapter.py
│   ├── ft_transformer_adapter.py
│   ├── node_adapter.py
│   ├── registry.py
│   └── tabnet_adapter.py
│
├── nn_models/                   # Deep learning model definitions
│   ├── ft_transformer_lib/      # FT-Transformer model
│   │   ├── ft_transformer.py
│   │   └── __init__.py
│   │
│   ├── node_lib/                # NODE model components
│   │   ├── arch.py
│   │   ├── data.py
│   │   ├── nn_utils.py
│   │   ├── odst.py
│   │   ├── trainer.py
│   │   ├── utils.py
│   │   └── __init__.py
│   │
│   ├── tabnet_lib/              # TabNet model components
│   │   ├── abstract_model.py
│   │   ├── augmentations.py
│   │   ├── callbacks.py
│   │   ├── metrics.py
│   │   ├── multitask_utils.py
│   │   ├── multiclass_utils.py
│   │   ├── pretraining.py
│   │   ├── pretraining_utils.py
│   │   ├── sparsemax.py
│   │   ├── tab_model.py
│   │   ├── tab_network.py
│   │   ├── utils.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── pages/                       # Streamlit UI pages
│   ├── 01_statistics.py
│   ├── 02_machine_learning.py
│   └── 03_deep_learning.py
│
├── pipelines/                   # Model training and orchestration
│   ├── __init__.py
│   ├── make_pipeline.py
│   ├── train_core.py
│   └── train_strategies/
│       ├── train_classical.py
│       ├── train_dnn.py
│       ├── train_ft.py
│       ├── train_node.py
│       └── train_tabnet.py
│
├── utils/                       # Common utility modules
│   ├── __init__.py
│   ├── config_parser.py
│   ├── logger.py
│   ├── metrics.py
│   └── optuna_search.py
│
├── venv/                        # Local virtual environment (ignored)
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── README.md
│
├── app.py                        # Main Streamlit entrypoint
├── data_processor.py             # Parsing and merging raw TXT/ZIP files
├── preprocess.py                 # Cleaning, normalization, imputation
├── models.py                     # Model definitions (Streamlit UI level)
├── statistics.py                 # Statistical visualizations
├── ui_shared.py                  # Shared UI components
└── run_training.py               # CLI entry for model training pipeline
```

---

## How the Application Works

### 1. Upload & Cleaning (app.py)
- Upload `.txt` files or `.zip` archives.
- Files are filtered for duplicates by name or hash.
- `data_processor.py` extracts, merges, and standardizes tables.
- `preprocess.py` cleans and imputes data using **KNNImputer**.
- Output is stored in `st.session_state` as `last_clean_df`.

### 2. Statistics Page (`01_statistics.py`)
- Displays summary statistics and correlation heatmaps.
- Provides dynamic visualization with Plotly and Streamlit.

### 3. Machine Learning Page (`02_machine_learning.py`)
- Trains and compares classical models (Random Forest, XGBoost, etc.).
- Displays performance metrics and exportable plots.

### 4. Deep Learning Page (`03_deep_learning.py`)
- Integrates neural models: TabNet, NODE, and FT-Transformer.
- Uses modules from `nn_models` and `pipelines/train_strategies`.
- Hyperparameter tuning via Optuna.

### 5. Training Pipelines
- Reusable training orchestration in `pipelines/train_core.py`.
- Each strategy defines its own training wrapper.

---

## Run Locally with Virtual Environment

```bash
python -m venv venv
.\venv\Scripts\Activate          # On Windows
# or
source venv/bin/activate         # On Linux/Mac

pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```

Then open: http://localhost:8501

---

## Run with Docker

```bash
docker build -t ehl-film-app .
docker run --rm -p 8501:8501 ehl-film-app
```

Then open: http://localhost:8501

---

## Dependencies

```
streamlit>=1.38
plotly>=5.23
pandas>=2.2
numpy>=1.26
scipy>=1.14
xlrd>=2.0.1
pyarrow>=17.0
joblib>=1.4
tqdm>=4.66
psutil>=5.9
matplotlib>=3.9
seaborn>=0.13
watchdog>=4.0
scikit-learn==1.7.1
tqdm>=4.66.0
optuna==4.4.0
PyYAML==6.0.2
xgboost>=2.0.0
lightgbm>=4.3.0
catboost>=1.2.2
joblib>=1.3.2
openpyxl==3.1.5
category_encoders==2.8.1
requests==2.32.4
tensorboardX==2.6.4
```

---

## Notes

- All models are modular and interchangeable.
- The project includes automatic duplicate filtering and auto-cleaning.
- The “Clear” button resets all uploads and cached states in Streamlit.

