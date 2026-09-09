# ArchForge

**An AI Engineering Bootstrap Framework that recommends and generates modular architectures for AI applications.**

ArchForge is an open-source Python framework designed to reduce the repetitive setup involved in building machine learning applications.

It analyzes a dataset, recommends an appropriate ML workflow, generates a modular project structure, and provides an end-to-end training and prediction pipeline.

## 🚀 Current Progress

ArchForge currently supports an end-to-end **Machine Learning workflow**:

```text
Dataset
   ↓
Data Ingestion
   ↓
Data Validation
   ↓
Dataset Profiling
   ↓
Target Selection
   ↓
Data Cleaning
   ↓
Data Transformation
   ↓
Model Selection
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Pushing
   ↓
Prediction
```

The generated project handles the major stages of an ML application while keeping the architecture modular and easy to extend.

## ✨ Current Features

### 📥 Data Ingestion

* Loads datasets from CSV files.
* Creates train/test datasets.
* Handles the basic dataset ingestion workflow automatically.

### 🔍 Data Validation

* Checks dataset structure and consistency.
* Detects missing values.
* Detects duplicate rows.
* Performs basic feature validation.

### 📊 Dataset Profiling

* Analyzes dataset size and feature composition.
* Identifies numerical and categorical features.
* Determines the ML problem type.
* Provides dataset safety/complexity information.
* Assists with target-column identification.

### 🎯 Target Selection

ArchForge supports both automatic and user-assisted target selection.

When the target column is unambiguous, ArchForge can identify it automatically.

When multiple columns are possible candidates, ArchForge presents the most likely candidates and allows the user to select the target explicitly.

This avoids unreliable target guesses on complex datasets.

### 🧹 Data Cleaning

* Handles missing numerical values.
* Handles missing categorical values.
* Removes inappropriate identifier columns.
* Removes constant columns where appropriate.
* Performs dataset cleaning while preserving the target column.

### 🔄 Data Transformation

ArchForge builds a reusable preprocessing pipeline:

* Numerical imputation
* Feature scaling
* Categorical imputation
* One-hot encoding
* Ordinal encoding support
* Unknown categorical values are handled safely

The fitted preprocessing pipeline is saved and reused during prediction.

### 🤖 Model Selection

ArchForge recommends models according to the detected problem type.

Currently supported models include:

**Regression**

* Linear Regression
* Ridge
* Lasso
* ElasticNet
* Decision Tree
* Random Forest
* Gradient Boosting
* AdaBoost
* KNN
* SVR

**Classification**

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* AdaBoost
* KNN
* SVC
* Gaussian Naive Bayes

Optional models such as **XGBoost** and **CatBoost** are supported when their dependencies are installed and are skipped gracefully when unavailable.

### 🧠 Model Training

The training component:

* Creates the selected model.
* Trains the model on the training dataset.
* Records model-training information.
* Keeps model training separate from model selection and evaluation.

### 📈 Model Evaluation

ArchForge evaluates trained models using appropriate metrics.

**Regression**

* MAE
* MSE
* RMSE
* R²

**Classification**

* Accuracy
* Weighted Precision
* Weighted Recall
* Weighted F1

The final evaluation is performed on the held-out test dataset.

### 🔬 Cross-Validation

ArchForge also supports cross-validation for model comparison.

The current workflow uses:

* **K-Fold Cross-Validation** for regression
* **Stratified K-Fold Cross-Validation** for classification
* 5 folds by default
* R² as the primary regression metric
* Weighted F1 as the primary classification metric

The best candidate model is selected based on cross-validation performance before final evaluation on the untouched test set.

### 💾 Artifact Management

Important artifacts are saved automatically:

```text
artifacts/
├── train.csv
├── test.csv
├── preprocessor.pkl
├── models/
│   ├── best_model.pkl
│   └── model_metadata.json
└── evaluation/
    ├── evaluation_report.json
    └── cross_validation_report.json
```

This allows the prediction pipeline to reuse the exact preprocessing and trained model.

### 🔮 Prediction Pipeline

The prediction pipeline:

* Loads the saved best model.
* Loads the saved preprocessing pipeline.
* Transforms new input using the same preprocessing logic.
* Generates predictions without retraining the model.

Training and prediction are therefore separated into independent workflows.

## 🧪 Testing & Validation

The core ML pipeline has been tested across several scenarios, including:

* Missing numerical and categorical values
* Duplicate rows
* Constant columns
* Identifier columns
* Small/unsafe datasets
* Regression datasets
* Classification datasets
* Unseen categorical values
* Optional ML dependencies
* Cross-validation and evaluation
* Artifact consistency
* Reproducibility

The reproducibility workflow was executed **five times with consistent results**.

## 🏗️ Architecture

ArchForge follows a modular component-based architecture.

```text
ArchForge/
│
├── app.py
│
├── src/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── dataset_profiler.py
│   ├── data_cleaning.py
│   ├── data_transformation.py
│   │
│   ├── model_selector.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   ├── model_pusher.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── templates/
│   ├── knowledge/
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
│
├── tests/
├── examples/
├── requirements.txt
└── README.md
```

The architecture is intentionally modular so individual components can be extended or replaced without redesigning the entire framework.

## ⚙️ Basic Workflow

A typical ArchForge workflow is:

```bash
python app.py
```

The user provides the project information and dataset, and ArchForge generates the required project structure and ML workflow.

The generated project can then be used to execute the training pipeline and produce a working prediction system.

## 🛠️ Technology Stack

* Python 3.11
* NumPy
* Pandas
* Scikit-learn
* Joblib
* YAML
* Streamlit
* Optional:

  * XGBoost
  * CatBoost

## 🎯 Project Goal

The long-term goal of ArchForge is to become a **bootstrap framework for AI engineering projects**, where developers can start with a dataset or application requirement and quickly obtain a clean, modular, production-oriented architecture.

Future versions may extend ArchForge beyond traditional ML into:

* Deep Learning
* Computer Vision
* NLP
* Advanced AI workflows
* Automated feature engineering
* More intelligent architecture recommendations
* Additional deployment workflows

## 📌 Current Status

**Status: Active Development — ML MVP**

The core Machine Learning pipeline is currently functional and has undergone testing across multiple data-quality, modeling, evaluation, dependency, and reproducibility scenarios.

ArchForge is still under active development, and the architecture and feature set will continue to evolve.

## 📄 License

License information will be added before the first public release.
