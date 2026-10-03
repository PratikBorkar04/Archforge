"""
text_preprocessing.py

Responsibilities
----------------
1. Load train/test/validation datasets.
2. Detect the most likely target column.
3. Ask the user to confirm or change the target.
4. Detect possible text columns.
5. Ask the user to confirm or change the text columns.
6. Combine selected text columns.
7. Clean the combined text.
8. Save processed datasets.

This module does NOT:
- Tokenize text.
- Apply TF-IDF.
- Train models.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


class NLPTextPreprocessing:

    TARGET_AUTO_THRESHOLD = 60
    TARGET_SCORE_GAP = 15

    TEXT_AUTO_THRESHOLD = 65
    TEXT_SCORE_GAP = 15

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
            self.artifacts_dir / "preprocessed"
        )

    # ==================================================
    # DATA LOADING
    # ==================================================

    def _load_dataset(
        self,
        filename: str,
    ) -> pd.DataFrame:

        path = self.artifacts_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        try:
            data = pd.read_csv(path)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to read dataset: {path}\n"
                f"Reason: {exc}"
            ) from exc

        if data.empty:
            raise ValueError(
                f"Dataset is empty: {filename}"
            )

        return data

    # ==================================================
    # TARGET COLUMN ANALYSIS
    # ==================================================

    @staticmethod
    def _is_identifier_like(
        series: pd.Series,
    ) -> bool:

        non_null = series.dropna()

        if non_null.empty:
            return False

        unique_ratio = (
            non_null.nunique()
            / len(non_null)
        )

        if unique_ratio >= 0.95:
            return True

        return False

    @staticmethod
    def _is_constant(
        series: pd.Series,
    ) -> bool:

        return series.dropna().nunique() <= 1

    @classmethod
    def _target_score(
        cls,
        data: pd.DataFrame,
        column: str,
    ) -> int:

        series = data[column]

        non_null = series.dropna()

        if non_null.empty:
            return -100

        if cls._is_constant(series):
            return -80

        score = 0

        unique_count = non_null.nunique()

        unique_ratio = (
            unique_count
            / len(non_null)
        )

        # ----------------------------------------------
        # Categorical / classification evidence
        # ----------------------------------------------

        if unique_count == 2:
            score += 35

        elif 3 <= unique_count <= 10:
            score += 25

        elif 11 <= unique_count <= 20:
            score += 10

        # ----------------------------------------------
        # Low cardinality
        # ----------------------------------------------

        if unique_ratio <= 0.05:
            score += 20

        elif unique_ratio <= 0.20:
            score += 10

        # ----------------------------------------------
        # Numeric target evidence
        # ----------------------------------------------

        if pd.api.types.is_numeric_dtype(series):

            score += 15

            if (
                pd.api.types.is_integer_dtype(series)
                and unique_count <= 20
            ):
                score += 10

        # ----------------------------------------------
        # Identifier penalty
        # ----------------------------------------------

        if cls._is_identifier_like(series):
            score -= 70

        # ----------------------------------------------
        # Name signal
        # Only a supporting signal.
        # ----------------------------------------------

        name = column.lower().strip()

        strong_names = (
            "target",
            "label",
            "class",
            "sentiment",
            "category",
            "outcome",
        )

        if name in strong_names:
            score += 20

        return max(score, 0)

    def _rank_target_columns(
        self,
        data: pd.DataFrame,
    ):

        scores = {}

        for column in data.columns:

            scores[column] = (
                self._target_score(
                    data,
                    column,
                )
            )

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked

    def _select_target_column(
        self,
        data: pd.DataFrame,
    ) -> str:

        ranked = self._rank_target_columns(data)

        if not ranked:
            raise ValueError(
                "No columns available for target detection."
            )

        best_column, best_score = ranked[0]

        second_score = (
            ranked[1][1]
            if len(ranked) > 1
            else 0
        )

        confident = (
            best_score >= self.TARGET_AUTO_THRESHOLD
            and (
                best_score - second_score
                >= self.TARGET_SCORE_GAP
            )
        )

        print("\nTarget column recommendation:")
        print(
            f"  {best_column} "
            f"({best_score}% confidence)"
        )

        while True:

            if confident:

                choice = input(
                    "\nUse this target column? [Y/n]: "
                ).strip().lower()

                if choice in ("", "y", "yes"):
                    return best_column

                if choice in ("n", "no"):
                    break

                print("Please enter Y or N.")

            else:

                print(
                    "\nArchForge could not confidently "
                    "select the target."
                )

                break

        # --------------------------------------------------
        # Manual target selection
        # --------------------------------------------------

        print("\nSelect target column:")

        for index, (column, score) in enumerate(
            ranked,
            start=1,
        ):
            print(
                f"  {index}. {column} "
                f"({score}% confidence)"
            )

        while True:

            choice = input(
                "\nEnter column number: "
            ).strip()

            try:
                index = int(choice)

            except ValueError:
                print("Enter a valid number.")
                continue

            if 1 <= index <= len(ranked):
                return ranked[index - 1][0]

            print("Invalid selection.")

    # ==================================================
    # TEXT COLUMN ANALYSIS
    # ==================================================

    @staticmethod
    def _text_score(
        data: pd.DataFrame,
        column: str,
    ) -> int:

        series = data[column]

        non_null = (
            series
            .dropna()
            .astype(str)
            .str.strip()
        )

        if non_null.empty:
            return 0

        # Text must be string/object-like.
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            return 0

        score = 20

        # ----------------------------------------------
        # Average text length
        # ----------------------------------------------

        average_length = (
            non_null.str.len().mean()
        )

        if average_length >= 100:
            score += 30

        elif average_length >= 50:
            score += 25

        elif average_length >= 20:
            score += 15

        elif average_length >= 10:
            score += 5

        # ----------------------------------------------
        # Average word count
        # ----------------------------------------------

        average_words = (
            non_null
            .str.split()
            .str.len()
            .mean()
        )

        if average_words >= 20:
            score += 30

        elif average_words >= 10:
            score += 25

        elif average_words >= 5:
            score += 15

        elif average_words >= 3:
            score += 5

        # ----------------------------------------------
        # Natural language characteristics
        # ----------------------------------------------

        space_ratio = (
            non_null.str.contains(
                r"\s",
                regex=True,
            ).mean()
        )

        if space_ratio >= 0.70:
            score += 10

        # ----------------------------------------------
        # Penalize ID-like columns
        # ----------------------------------------------

        unique_ratio = (
            non_null.nunique()
            / len(non_null)
        )

        if unique_ratio >= 0.99 and average_length < 15:
            score -= 40

        return max(
            min(score, 100),
            0,
        )

    def _rank_text_columns(
        self,
        data: pd.DataFrame,
        target_column: str,
    ):

        scores = {}

        for column in data.columns:

            if column == target_column:
                continue

            scores[column] = (
                self._text_score(
                    data,
                    column,
                )
            )

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked

    def _select_text_columns(
        self,
        data: pd.DataFrame,
        target_column: str,
    ) -> list[str]:

        ranked = self._rank_text_columns(
            data,
            target_column,
        )

        candidates = [
            item
            for item in ranked
            if item[1] >= self.TEXT_AUTO_THRESHOLD
        ]

        if not candidates:

            raise ValueError(
                "\nNo sufficiently strong text "
                "column detected."
            )

        # --------------------------------------------------
        # Display recommendation
        # --------------------------------------------------

        print("\nText columns detected:")

        for index, (column, score) in enumerate(
            candidates,
            start=1,
        ):
            print(
                f"  {index}. {column} "
                f"({score}% confidence)"
            )

        print(
            "\nArchForge recommends combining "
            f"{len(candidates)} text column(s)."
        )

        while True:

            choice = input(
                "Use these text columns? [Y/n]: "
            ).strip().lower()

            if choice in ("", "y", "yes"):
                return [
                    column
                    for column, _ in candidates
                ]

            if choice in ("n", "no"):
                break

            print("Please enter Y or N.")

        # --------------------------------------------------
        # Manual selection
        # --------------------------------------------------

        print("\nAll possible text columns:")

        for index, (column, score) in enumerate(
            ranked,
            start=1,
        ):
            print(
                f"  {index}. {column} "
                f"({score}% confidence)"
            )

        print(
            "\nEnter column numbers separated "
            "by commas."
        )

        while True:

            choice = input(
                "Example: 1,2: "
            ).strip()

            try:

                numbers = [
                    int(value.strip())
                    for value in choice.split(",")
                ]

            except ValueError:

                print(
                    "Enter numbers separated by commas."
                )

                continue

            if not numbers:
                continue

            if len(set(numbers)) != len(numbers):
                print(
                    "Do not select the same column twice."
                )
                continue

            if any(
                number < 1
                or number > len(ranked)
                for number in numbers
            ):
                print("Invalid column number.")
                continue

            selected = [
                ranked[number - 1][0]
                for number in numbers
            ]

            return selected

    # ==================================================
    # COMBINE TEXT COLUMNS
    # ==================================================

    @staticmethod
    def _combine_text_columns(
        data: pd.DataFrame,
        text_columns: list[str],
    ) -> pd.DataFrame:

        processed = data.copy()

        processed["combined_text"] = (
            processed[text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(
                r"\s+",
                " ",
                regex=True,
            )
            .str.strip()
            .str.lower()
        )

        processed = processed.drop(
            columns=text_columns
        )

        return processed

    # ==================================================
    # SAVE DATASET
    # ==================================================

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

        return path

    # ==================================================
    # MAIN
    # ==================================================

    def initiate_text_preprocessing(self):

        print(
            "\n" + "=" * 60
        )
        print(
            "ARCHFORGE - NLP COLUMN ANALYSIS"
        )
        print(
            "=" * 60
        )

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

        # ==================================================
        # TARGET SELECTION
        # ==================================================

        target_column = (
            self._select_target_column(
                train_data
            )
        )

        print(
            f"\nSelected target: {target_column}"
        )

        # ==================================================
        # TEXT COLUMN SELECTION
        # ==================================================

        text_columns = (
            self._select_text_columns(
                train_data,
                target_column,
            )
        )

        print(
            "\nSelected text columns:"
        )

        for column in text_columns:
            print(f"  • {column}")

        # ==================================================
        # SAFETY CHECK
        # ==================================================

        if target_column in text_columns:

            raise RuntimeError(
                "Safety check failed: target column "
                "was selected as a text column."
            )

        # ==================================================
        # PROCESS DATASETS
        # ==================================================

        processed_train = (
            self._combine_text_columns(
                train_data,
                text_columns,
            )
        )

        processed_test = (
            self._combine_text_columns(
                test_data,
                text_columns,
            )
        )

        processed_validation = None

        if validation_data is not None:

            processed_validation = (
                self._combine_text_columns(
                    validation_data,
                    text_columns,
                )
            )

        # ==================================================
        # SAVE
        # ==================================================

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

        # ==================================================
        # SUMMARY
        # ==================================================

        print(
            "\n" + "-" * 60
        )

        print(
            "NLP preprocessing completed."
        )

        print(
            f"Target       : {target_column}"
        )

        print(
            f"Text columns : {len(text_columns)}"
        )

        print(
            f"Train        : {train_path}"
        )

        print(
            f"Test         : {test_path}"
        )

        if validation_path_saved:
            print(
                f"Validation   : {validation_path_saved}"
            )

        print(
            "-" * 60
        )

        return (
            processed_train,
            processed_validation,
            processed_test,
            text_columns,
            target_column,
        )