
"""
model_trainer.py

Responsibilities:
1. Load TF-IDF matrices and target labels.
2. Train multiple lightweight NLP models.
3. Compare models using weighted F1-score.
4. Select the best model.
5. Retrain the selected model on full training data.
6. Save the trained model and training report.
"""

from pathlib import Path
import json

import joblib
import pandas as pd

from scipy.sparse import load_npz

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class NLPModelTrainer:

    def __init__(self, project_root: Path) -> None:

        self.project_root = Path(project_root).resolve()

        self.transformed_dir = (
            self.project_root / "artifacts" / "transformed"
        )

        self.models_dir = (
            self.project_root / "artifacts" / "models"
        )

        self.best_model_path = (
            self.models_dir / "best_model.pkl"
        )

        self.report_path = (
            self.models_dir / "training_report.json"
        )

        self.random_state = 42

    def _load_training_data(self):

        X_train = load_npz(
            self.transformed_dir / "X_train.npz"
        )

        y_train = pd.read_csv(
            self.transformed_dir / "y_train.csv"
        ).squeeze("columns")

        if y_train.isnull().any():
            raise ValueError(
                "Training target contains missing values."
            )

        # Sparse matrices: use shape[0], not len().
        if X_train.shape[0] != len(y_train):
            raise ValueError(
                "Training feature and target row counts do not match."
            )

        return X_train, y_train

    def _load_validation_data(self):

        X_path = self.transformed_dir / "X_validation.npz"
        y_path = self.transformed_dir / "y_validation.csv"

        if X_path.exists() and y_path.exists():

            X_validation = load_npz(X_path)

            y_validation = pd.read_csv(
                y_path
            ).squeeze("columns")

            if X_validation.shape[0] != len(y_validation):
                raise ValueError(
                    "Validation feature and target row counts do not match."
                )

            return X_validation, y_validation

        if X_path.exists() != y_path.exists():
            raise ValueError(
                "Incomplete validation data. "
                "Both validation files must exist."
            )

        return None, None

    @staticmethod
    def _get_models():

        return {
            "Multinomial Naive Bayes": MultinomialNB(),

            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),

            "Linear SVM": LinearSVC(
                random_state=42,
                max_iter=2000,
            ),
        }

    def initiate_model_training(self):

        print("\nNLP MODEL TRAINING")

        X_train, y_train = self._load_training_data()

        X_validation, y_validation = (
            self._load_validation_data()
        )

        if X_validation is None:

            print(
                "No validation dataset found. "
                "Creating internal 80/20 training split."
            )

            train_indices, validation_indices = train_test_split(
                list(range(X_train.shape[0])),
                test_size=0.20,
                random_state=self.random_state,
                stratify=y_train,
            )

            X_fit = X_train[train_indices]
            X_validation = X_train[validation_indices]

            y_fit = y_train.iloc[train_indices]
            y_validation = y_train.iloc[validation_indices]

            validation_method = "internal_80_20_split"

        else:

            print("Using provided validation dataset.")

            X_fit = X_train
            y_fit = y_train

            validation_method = "provided_validation"

        models = self._get_models()
        results = {}

        print("\nComparing NLP models...")

        for model_name, model in models.items():

            print(f"\nTraining: {model_name}")

            model.fit(X_fit, y_fit)

            predictions = model.predict(X_validation)

            accuracy = accuracy_score(
                y_validation,
                predictions,
            )

            f1 = f1_score(
                y_validation,
                predictions,
                average="weighted",
                zero_division=0,
            )

            results[model_name] = {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
            }

            print(f"Accuracy : {accuracy:.4f}")
            print(f"F1-score : {f1:.4f}")

        best_model_name = max(
            results,
            key=lambda name: results[name]["f1_score"],
        )

        print(f"\nBest model: {best_model_name}")

        # Retrain selected model on all training data.
        best_model = self._get_models()[best_model_name]

        print("Retraining best model on complete training data...")

        best_model.fit(X_train, y_train)

        self.models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            best_model,
            self.best_model_path,
        )

        report = {
            "best_model": best_model_name,
            "selection_metric": "weighted_f1_score",
            "validation_method": validation_method,
            "models": results,
        }

        with self.report_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(report, file, indent=4)

        print("\nModel training completed.")
        print(f"Best model : {best_model_name}")
        print(f"Saved at   : {self.best_model_path}")

        return best_model, best_model_name, results
