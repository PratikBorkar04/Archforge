"""
text_preprocessing.py

Generic text preprocessing module for ArchForge NLP projects.

Responsibilities
----------------
1. Validate the selected text column.
2. Handle missing text values.
3. Convert text to string.
4. Normalize whitespace.
5. Convert text to lowercase.
6. Optionally remove unwanted characters.
7. Save the processed datasets.

This module does NOT:
- Detect the text column.
- Detect the target column.
- Perform tokenization.
- Apply TF-IDF.
- Train a model.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


class NLPTextPreprocessing:

    def __init__(
        self,
        project_root: Path,
    ) -> None:

        self.project_root = Path(
            project_root
        ).resolve()

        self.artifacts_dir = (
            self.project_root / "artifacts"
        )

        self.preprocessing_dir = (
            self.artifacts_dir
            / "preprocessed"
        )

    # --------------------------------------------------
    # Dataset loading
    # --------------------------------------------------

    def _load_dataset(
        self,
        filename: str,
    ) -> pd.DataFrame:

        path = (
            self.artifacts_dir / filename
        )

        if not path.exists():

            raise FileNotFoundError(
                f"\nDataset not found:\n{path}"
            )

        try:

            data = pd.read_csv(path)

        except Exception as exc:

            raise RuntimeError(
                f"\nUnable to read dataset:\n"
                f"{path}\n\n"
                f"Reason: {exc}"
            ) from exc

        if data.empty:

            raise ValueError(
                f"\nDataset is empty:\n"
                f"{path.name}"
            )

        return data

    # --------------------------------------------------
    # Text column validation
    # --------------------------------------------------

    @staticmethod
    def _validate_text_column(
        data: pd.DataFrame,
        text_column: str,
    ) -> None:

        if text_column not in data.columns:

            raise ValueError(
                f"\nText column '{text_column}' "
                f"was not found.\n\n"
                f"Available columns:\n"
                f"{list(data.columns)}"
            )

    # --------------------------------------------------
    # Text cleaning
    # --------------------------------------------------

    @staticmethod
    def _clean_text(
        value,
    ) -> str:

        if pd.isna(value):

            return ""

        text = str(value)

        # Convert to lowercase
        text = text.lower()

        # Normalize whitespace
        text = " ".join(
            text.split()
        )

        return text

    def _process_dataset(
        self,
        data: pd.DataFrame,
        text_column: str,
    ) -> pd.DataFrame:

        self._validate_text_column(
            data,
            text_column,
        )

        processed_data = data.copy()

        processed_data[text_column] = (
            processed_data[text_column]
            .apply(self._clean_text)
        )

        return processed_data

    # --------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------

    def _save_dataset(
        self,
        data: pd.DataFrame,
        filename: str,
    ) -> Path:

        self.preprocessing_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            self.preprocessing_dir
            / filename
        )

        data.to_csv(
            path,
            index=False,
        )

        print(
            f"Saved: {path}"
        )

        return path

    # --------------------------------------------------
    # Main preprocessing
    # --------------------------------------------------

    def initiate_text_preprocessing(
        self,
        text_column: str,
    ):

        print(
            "\n" + "=" * 60
        )

        print(
            "NLP TEXT PREPROCESSING"
        )

        print(
            "=" * 60
        )

        print(
            f"\nText column: {text_column}"
        )

        # --------------------------------------------------
        # Load datasets
        # --------------------------------------------------

        train_data = self._load_dataset(
            "train.csv"
        )

        test_data = self._load_dataset(
            "test.csv"
        )

        validation_path = (
            self.artifacts_dir
            / "validation.csv"
        )

        validation_data: Optional[
            pd.DataFrame
        ] = None

        if validation_path.exists():

            validation_data = (
                self._load_dataset(
                    "validation.csv"
                )
            )

        print(
            "\nDatasets loaded successfully."
        )

        # --------------------------------------------------
        # Process datasets
        # --------------------------------------------------

        print(
            "\nProcessing training data..."
        )

        processed_train = (
            self._process_dataset(
                train_data,
                text_column,
            )
        )

        print(
            "Processing testing data..."
        )

        processed_test = (
            self._process_dataset(
                test_data,
                text_column,
            )
        )

        processed_validation = None

        if validation_data is not None:

            print(
                "Processing validation data..."
            )

            processed_validation = (
                self._process_dataset(
                    validation_data,
                    text_column,
                )
            )

        # --------------------------------------------------
        # Save datasets
        # --------------------------------------------------

        train_path = self._save_dataset(
            processed_train,
            "train.csv",
        )

        test_path = self._save_dataset(
            processed_test,
            "test.csv",
        )

        validation_path_saved = None

        if processed_validation is not None:

            validation_path_saved = (
                self._save_dataset(
                    processed_validation,
                    "validation.csv",
                )
            )

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "NLP TEXT PREPROCESSING SUMMARY"
        )

        print(
            "=" * 60
        )

        print(
            f"\nTraining samples   : "
            f"{len(processed_train)}"
        )

        print(
            f"Testing samples    : "
            f"{len(processed_test)}"
        )

        validation_count = (
            len(processed_validation)
            if processed_validation is not None
            else "Not provided"
        )

        print(
            f"Validation samples : {validation_count}"
        )

        print(
            f"\nProcessed datasets:"
        )

        print(
            f"  → {train_path}"
        )

        if validation_path_saved:

            print(
                f"  → {validation_path_saved}"
            )

        print(
            f"  → {test_path}"
        )

        print(
            "\nNLP text preprocessing "
            "completed successfully."
        )

        print(
            "=" * 60
        )

        return (
            processed_train,
            processed_validation,
            processed_test,
        )