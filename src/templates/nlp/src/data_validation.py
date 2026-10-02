"""
data_validation.py

Generic data validation module for ArchForge NLP projects.

Responsibilities
----------------
1. Verify train/test/validation datasets exist.
2. Read and validate CSV files.
3. Check whether datasets are empty.
4. Check schema consistency.
5. Detect duplicate rows.
6. Detect missing values.
7. Detect completely empty columns.
8. Detect blank values in string/object columns.
9. Save a validation report.

This module does NOT:
- Detect the text column.
- Detect the target column.
- Clean or tokenize text.
- Apply TF-IDF.
- Train models.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


class NLPDataValidation:

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()

        self.artifacts_dir = (
            self.project_root / "artifacts"
        )

        self.validation_dir = (
            self.artifacts_dir / "validation"
        )

    # --------------------------------------------------
    # File handling
    # --------------------------------------------------

    def _get_dataset_path(
        self,
        filename: str,
    ) -> Path:

        path = self.artifacts_dir / filename

        if not path.exists():

            raise FileNotFoundError(
                f"\nDataset not found:\n{path}"
            )

        return path

    def _read_dataset(
        self,
        path: Path,
    ) -> pd.DataFrame:

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
    # Basic validation
    # --------------------------------------------------

    @staticmethod
    def _validate_columns(
        data: pd.DataFrame,
        dataset_name: str,
    ) -> None:

        if len(data.columns) == 0:

            raise ValueError(
                f"\n{dataset_name} contains no columns."
            )

        duplicated_columns = (
            data.columns[
                data.columns.duplicated()
            ].tolist()
        )

        if duplicated_columns:

            raise ValueError(
                f"\nDuplicate column names detected "
                f"in {dataset_name}:\n"
                f"{duplicated_columns}"
            )

    @staticmethod
    def _find_duplicate_rows(
        data: pd.DataFrame,
    ) -> int:

        return int(
            data.duplicated().sum()
        )

    @staticmethod
    def _find_missing_values(
        data: pd.DataFrame,
    ) -> dict:

        missing = (
            data.isnull()
            .sum()
            .to_dict()
        )

        return {
            column: int(count)
            for column, count in missing.items()
            if count > 0
        }

    @staticmethod
    def _find_blank_values(
        data: pd.DataFrame,
    ) -> dict:

        blank_values = {}

        object_columns = (
            data.select_dtypes(
                include=["object", "string"]
            ).columns
        )

        for column in object_columns:

            blank_count = int(
                data[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            )

            if blank_count > 0:

                blank_values[column] = (
                    blank_count
                )

        return blank_values

    @staticmethod
    def _find_empty_columns(
        data: pd.DataFrame,
    ) -> list:

        empty_columns = []

        for column in data.columns:

            if data[column].isnull().all():

                empty_columns.append(column)

        return empty_columns

    # --------------------------------------------------
    # Schema validation
    # --------------------------------------------------

    @staticmethod
    def _validate_schema(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
    ) -> None:

        train_columns = list(
            train_data.columns
        )

        test_columns = list(
            test_data.columns
        )

        if train_columns != test_columns:

            raise ValueError(
                "\nTrain/test schema mismatch.\n\n"
                f"Train columns:\n"
                f"{train_columns}\n\n"
                f"Test columns:\n"
                f"{test_columns}"
            )

        if validation_data is not None:

            validation_columns = list(
                validation_data.columns
            )

            if (
                train_columns
                != validation_columns
            ):

                raise ValueError(
                    "\nTrain/validation schema mismatch.\n\n"
                    f"Train columns:\n"
                    f"{train_columns}\n\n"
                    f"Validation columns:\n"
                    f"{validation_columns}"
                )

    # --------------------------------------------------
    # Dataset inspection
    # --------------------------------------------------

    def _inspect_dataset(
        self,
        data: pd.DataFrame,
        dataset_name: str,
    ) -> dict:

        self._validate_columns(
            data,
            dataset_name,
        )

        duplicate_rows = (
            self._find_duplicate_rows(data)
        )

        missing_values = (
            self._find_missing_values(data)
        )

        blank_values = (
            self._find_blank_values(data)
        )

        empty_columns = (
            self._find_empty_columns(data)
        )

        report = {

            "dataset": dataset_name,

            "rows": int(len(data)),

            "columns": int(len(data.columns)),

            "column_names": list(
                data.columns
            ),

            "duplicate_rows": duplicate_rows,

            "missing_values": missing_values,

            "blank_values": blank_values,

            "empty_columns": empty_columns,

        }

        return report

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    def _print_dataset_report(
        self,
        report: dict,
    ) -> None:

        print(
            "\n" + "-" * 60
        )

        print(
            f"{report['dataset']} DATASET"
        )

        print(
            "-" * 60
        )

        print(
            f"Rows              : "
            f"{report['rows']}"
        )

        print(
            f"Columns           : "
            f"{report['columns']}"
        )

        print(
            f"Duplicate rows    : "
            f"{report['duplicate_rows']}"
        )

        print(
            f"Missing values    : "
            f"{sum(report['missing_values'].values())}"
        )

        print(
            f"Blank values      : "
            f"{sum(report['blank_values'].values())}"
        )

        print(
            f"Empty columns     : "
            f"{len(report['empty_columns'])}"
        )

        if report["missing_values"]:

            print(
                "\nMissing values by column:"
            )

            for column, count in (
                report["missing_values"].items()
            ):

                print(
                    f"  • {column}: {count}"
                )

        if report["blank_values"]:

            print(
                "\nBlank values by column:"
            )

            for column, count in (
                report["blank_values"].items()
            ):

                print(
                    f"  • {column}: {count}"
                )

        if report["empty_columns"]:

            print(
                "\nCompletely empty columns:"
            )

            for column in (
                report["empty_columns"]
            ):

                print(
                    f"  • {column}"
                )

    # --------------------------------------------------
    # Save report
    # --------------------------------------------------

    def _save_report(
        self,
        reports: list,
    ) -> Path:

        self.validation_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path = (
            self.validation_dir
            / "validation_report.json"
        )

        import json

        with report_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                reports,
                file,
                indent=4,
                default=str,
            )

        return report_path

    # --------------------------------------------------
    # Main validation
    # --------------------------------------------------

    def initiate_data_validation(self):

        print(
            "\n" + "=" * 60
        )

        print(
            "NLP DATA VALIDATION"
        )

        print(
            "=" * 60
        )

        # --------------------------------------------------
        # Locate datasets
        # --------------------------------------------------

        train_path = (
            self._get_dataset_path(
                "train.csv"
            )
        )

        test_path = (
            self._get_dataset_path(
                "test.csv"
            )
        )

        validation_path = (
            self.artifacts_dir
            / "validation.csv"
        )

        # --------------------------------------------------
        # Read datasets
        # --------------------------------------------------

        train_data = (
            self._read_dataset(
                train_path
            )
        )

        test_data = (
            self._read_dataset(
                test_path
            )
        )

        validation_data = None

        if validation_path.exists():

            validation_data = (
                self._read_dataset(
                    validation_path
                )
            )

        print(
            "\nDatasets loaded successfully."
        )

        # --------------------------------------------------
        # Schema validation
        # --------------------------------------------------

        self._validate_schema(
            train_data,
            test_data,
            validation_data,
        )

        print(
            "\nSchema consistency check passed."
        )

        # --------------------------------------------------
        # Inspect datasets
        # --------------------------------------------------

        reports = []

        train_report = (
            self._inspect_dataset(
                train_data,
                "TRAIN",
            )
        )

        reports.append(
            train_report
        )

        self._print_dataset_report(
            train_report
        )

        test_report = (
            self._inspect_dataset(
                test_data,
                "TEST",
            )
        )

        reports.append(
            test_report
        )

        self._print_dataset_report(
            test_report
        )

        if validation_data is not None:

            validation_report = (
                self._inspect_dataset(
                    validation_data,
                    "VALIDATION",
                )
            )

            reports.append(
                validation_report
            )

            self._print_dataset_report(
                validation_report
            )

        # --------------------------------------------------
        # Save report
        # --------------------------------------------------

        report_path = (
            self._save_report(
                reports
            )
        )

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "NLP DATA VALIDATION SUMMARY"
        )

        print(
            "=" * 60
        )

        print(
            f"\nValidation report:"
            f"\n{report_path}"
        )

        print(
            "\nNLP data validation "
            "completed successfully."
        )

        print(
            "=" * 60
        )

        return (
            train_data,
            validation_data,
            test_data,
            reports,
        )