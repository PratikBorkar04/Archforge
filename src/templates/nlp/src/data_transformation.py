
"""
data_transformation.py

Responsibilities:
1. Load tokenized train/test/validation datasets.
2. Apply TF-IDF vectorization.
3. Fit vectorizer only on training data.
4. Transform test and validation data.
5. Save vectorizer and transformed datasets.
6. Extract target labels using the dynamically detected target column.
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


class NLPDataTransformation:

    def __init__(
        self,
        project_root: Path,
        target_column: str,
    ) -> None:

        self.project_root = Path(project_root).resolve()
        self.target_column = target_column

        if not self.target_column:
            raise ValueError(
                "A valid target column must be provided."
            )

        self.input_dir = (
            self.project_root / "artifacts" / "tokenized"
        )

        self.output_dir = (
            self.project_root / "artifacts" / "transformed"
        )

        self.vectorizer_path = (
            self.output_dir / "tfidf_vectorizer.pkl"
        )

        self.max_features = 5000

    def _load_dataset(self, filename: str) -> pd.DataFrame:

        path = self.input_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Tokenized dataset not found: {path}"
            )

        data = pd.read_csv(path)

        required_columns = {"combined_text", self.target_column}

        missing_columns = required_columns - set(data.columns)

        if missing_columns:
            raise ValueError(
                f"Required columns missing in {filename}: "
                f"{sorted(missing_columns)}"
            )

        return data

    def _extract_text_and_target(self, data: pd.DataFrame):

        text = data["combined_text"].fillna("").astype(str)

        target = data[self.target_column]

        return text, target

    def initiate_data_transformation(self):

        print("\nNLP DATA TRANSFORMATION (TF-IDF)")
        print(f"Target column: {self.target_column}")

        train_data = self._load_dataset("train.csv")
        test_data = self._load_dataset("test.csv")

        validation_path = self.input_dir / "validation.csv"
        validation_data: Optional[pd.DataFrame] = None

        if validation_path.exists():
            validation_data = self._load_dataset("validation.csv")

        X_train_text, y_train = self._extract_text_and_target(
            train_data
        )

        X_test_text, y_test = self._extract_text_and_target(
            test_data
        )

        X_validation_text = None
        y_validation = None

        if validation_data is not None:
            X_validation_text, y_validation = (
                self._extract_text_and_target(validation_data)
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )

        # Fit ONLY on training text.
        X_train = vectorizer.fit_transform(X_train_text)

        # Transform other splits using the training vocabulary.
        X_test = vectorizer.transform(X_test_text)

        X_validation = None

        if X_validation_text is not None:
            X_validation = vectorizer.transform(
                X_validation_text
            )

        # Save vectorizer.
        joblib.dump(vectorizer, self.vectorizer_path)

        # Save sparse feature matrices.
        save_npz(self.output_dir / "X_train.npz", X_train)
        save_npz(self.output_dir / "X_test.npz", X_test)

        if X_validation is not None:
            save_npz(
                self.output_dir / "X_validation.npz",
                X_validation,
            )

        # Save target labels separately.
        y_train.to_csv(
            self.output_dir / "y_train.csv",
            index=False,
        )

        y_test.to_csv(
            self.output_dir / "y_test.csv",
            index=False,
        )

        if y_validation is not None:
            y_validation.to_csv(
                self.output_dir / "y_validation.csv",
                index=False,
            )

        print("\nTF-IDF transformation completed.")
        print(f"Training matrix   : {X_train.shape}")
        print(f"Testing matrix    : {X_test.shape}")

        if X_validation is not None:
            print(f"Validation matrix : {X_validation.shape}")

        print(f"Vocabulary size   : {len(vectorizer.vocabulary_)}")
        print(f"Vectorizer saved  : {self.vectorizer_path}")

        return (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test,
            vectorizer,
        )
