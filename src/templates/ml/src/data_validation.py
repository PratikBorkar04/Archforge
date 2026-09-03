"""
data_validation.py

Automatic data validation module for ArchForge.

Responsibilities
----------------
1. Check train/test/validation datasets.
2. Check column consistency.
3. Detect missing values.
4. Detect duplicate rows.
5. Validate target column when explicitly known.
6. Identify numerical and categorical columns.
7. Save a validation report.

Important
---------
Data Validation does NOT remove or impute missing values.

Missing values are handled by DataCleaning.

If the target column is not known yet, target validation is
deferred to the Dataset Profiler.
"""

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


class DataValidation:
    """
    Generic automatic dataset validation component.
    """

    def __init__(
        self,
        project_root: Path,
        target_column: Optional[str] = None,
    ) -> None:

        self.project_root = Path(
            project_root
        ).resolve()

        self.artifacts_dir = (
            self.project_root
            / "artifacts"
        )

        self.report_dir = (
            self.project_root
            / "artifacts"
            / "validation"
        )

        self.target_column = target_column

    # ============================================================
    # Basic Dataset Checks
    # ============================================================

    @staticmethod
    def _check_empty_dataset(
        data: pd.DataFrame,
        name: str,
    ) -> None:

        if data is None:

            raise ValueError(
                f"{name} dataset cannot be None."
            )

        if data.empty:

            raise ValueError(
                f"{name} dataset is empty."
            )

    # ============================================================
    # Missing Values
    # ============================================================

    @staticmethod
    def _check_missing_values(
        data: pd.DataFrame,
        name: str,
    ) -> Dict[str, int]:
        """
        Detect missing values in every column.

        IMPORTANT:
        Missing values are warnings only.

        DataCleaning is responsible for handling them.
        """

        missing = (
            data.isnull()
            .sum()
            .to_dict()
        )

        missing = {
            column: int(count)
            for column, count
            in missing.items()
            if count > 0
        }

        if missing:

            print(
                f"\nWarning: Missing values "
                f"detected in {name}:"
            )

            for column, count in (
                missing.items()
            ):

                percentage = (
                    count
                    / len(data)
                    * 100
                    if len(data) > 0
                    else 0
                )

                print(
                    f"  • {column}: "
                    f"{count} "
                    f"({percentage:.2f}%)"
                )

        else:

            print(
                f"\n{name}: "
                "No missing values detected."
            )

        return missing

    # ============================================================
    # Duplicate Rows
    # ============================================================

    @staticmethod
    def _check_duplicates(
        data: pd.DataFrame,
        name: str,
    ) -> int:
        """
        Count duplicate rows.
        """

        duplicates = int(
            data.duplicated()
            .sum()
        )

        if duplicates:

            percentage = (
                duplicates
                / len(data)
                * 100
                if len(data) > 0
                else 0
            )

            print(
                f"\nWarning: {duplicates} "
                f"duplicate rows detected "
                f"in {name} "
                f"({percentage:.2f}%)."
            )

        else:

            print(
                f"\n{name}: "
                "No duplicate rows detected."
            )

        return duplicates

    # ============================================================
    # Column Consistency
    # ============================================================

    @staticmethod
    def _check_column_consistency(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        validation_data: Optional[
            pd.DataFrame
        ] = None,
    ) -> None:
        """
        Ensure train/test/validation datasets
        contain the same columns.
        """

        train_columns = set(
            train_data.columns
        )

        test_columns = set(
            test_data.columns
        )

        if train_columns != test_columns:

            missing_in_test = (
                train_columns
                - test_columns
            )

            extra_in_test = (
                test_columns
                - train_columns
            )

            raise ValueError(
                "\nTraining and testing "
                "columns do not match.\n\n"
                f"Missing in test: "
                f"{missing_in_test}\n"
                f"Extra in test: "
                f"{extra_in_test}"
            )

        if validation_data is not None:

            validation_columns = set(
                validation_data.columns
            )

            if (
                train_columns
                != validation_columns
            ):

                missing_in_validation = (
                    train_columns
                    - validation_columns
                )

                extra_in_validation = (
                    validation_columns
                    - train_columns
                )

                raise ValueError(
                    "\nTraining and validation "
                    "columns do not match.\n\n"
                    f"Missing in validation: "
                    f"{missing_in_validation}\n"
                    f"Extra in validation: "
                    f"{extra_in_validation}"
                )

        print(
            "\nColumn consistency check: PASSED"
        )

    # ============================================================
    # Target Validation
    # ============================================================

    def _validate_target(
        self,
        train_data: pd.DataFrame,
    ) -> Optional[str]:
        """
        Validate the target column only when it is known.

        IMPORTANT
        ---------
        This method does NOT automatically select the last column.

        If target_column is None, target validation is deferred
        to DatasetProfiler.

        Missing target values are NOT treated as a validation
        failure because DataCleaning is responsible for removing
        rows with missing target values.
        """

        # --------------------------------------------------------
        # Target not known yet
        # --------------------------------------------------------

        if self.target_column is None:

            print(
                "\nTarget column not specified."
            )

            print(
                "Target validation deferred "
                "to Dataset Profiler."
            )

            return None

        # --------------------------------------------------------
        # Check target exists
        # --------------------------------------------------------

        if (
            self.target_column
            not in train_data.columns
        ):

            raise ValueError(
                f"\nTarget column "
                f"'{self.target_column}' "
                "was not found in training data."
            )

        target = self.target_column

        # --------------------------------------------------------
        # Missing target values
        # --------------------------------------------------------

        missing_target = int(
            train_data[target]
            .isnull()
            .sum()
        )

        if missing_target:

            percentage = (
                missing_target
                / len(train_data)
                * 100
                if len(train_data) > 0
                else 0
            )

            print(
                f"\nWarning: Target column "
                f"'{target}' contains "
                f"{missing_target} missing values "
                f"({percentage:.2f}%)."
            )

            print(
                "These rows will be handled "
                "by DataCleaning."
            )

        else:

            print(
                f"\nTarget column '{target}': "
                "No missing values detected."
            )

        # --------------------------------------------------------
        # Check whether target has any valid values
        # --------------------------------------------------------

        valid_target = (
            train_data[target]
            .dropna()
        )

        if valid_target.empty:

            raise ValueError(
                f"\nTarget column "
                f"'{target}' contains "
                "no valid values."
            )

        # --------------------------------------------------------
        # Unique value check
        # --------------------------------------------------------

        unique_values = (
            valid_target.nunique()
        )

        if unique_values <= 1:

            raise ValueError(
                f"\nTarget column "
                f"'{target}' contains "
                "only one unique value "
                "after excluding missing values."
            )

        print(
            f"\nTarget column: {target}"
        )

        return target

    # ============================================================
    # Feature Analysis
    # ============================================================

    @staticmethod
    def _analyze_features(
        train_data: pd.DataFrame,
        target_column: Optional[str],
    ) -> Dict[str, list]:
        """
        Identify numerical and categorical features.

        If target is not known yet, all columns are analyzed
        as dataset columns. Final feature analysis happens
        after target selection.
        """

        if target_column is not None:

            features = train_data.drop(
                columns=[target_column]
            )

        else:

            features = train_data.copy()

        numerical_columns = (
            features
            .select_dtypes(
                include=["number"]
            )
            .columns
            .tolist()
        )

        categorical_columns = (
            features
            .select_dtypes(
                exclude=["number"]
            )
            .columns
            .tolist()
        )

        print(
            "\nFeature analysis:"
        )

        print(
            f"  Numerical features   : "
            f"{len(numerical_columns)}"
        )

        print(
            f"  Categorical features : "
            f"{len(categorical_columns)}"
        )

        return {
            "numerical": numerical_columns,
            "categorical": categorical_columns,
        }

    # ============================================================
    # Dataset Statistics
    # ============================================================

    @staticmethod
    def _dataset_statistics(
        data: pd.DataFrame,
    ) -> Dict[str, Any]:

        return {
            "rows": len(data),
            "columns": len(data.columns),
            "memory_usage": int(
                data.memory_usage(
                    deep=True
                ).sum()
            ),
        }

    # ============================================================
    # Save Validation Report
    # ============================================================

    def _save_report(
        self,
        report: Dict[str, Any],
    ) -> Path:
        """
        Save validation report as JSON.
        """

        import json

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path = (
            self.report_dir
            / "validation_report.json"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                default=str,
            )

        return report_path

    # ============================================================
    # Main Validation
    # ============================================================

    def initiate_data_validation(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        validation_data: Optional[
            pd.DataFrame
        ] = None,
    ) -> Dict[str, Any]:
        """
        Run complete automatic validation.

        Missing values are detected but NOT repaired.

        Target selection is performed only when a target was
        explicitly supplied. Otherwise target detection is
        deferred to DatasetProfiler.
        """

        print(
            "\n[2/6] Data Validation"
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "DATA VALIDATION"
        )

        print(
            "=" * 60
        )

        # --------------------------------------------------------
        # Basic checks
        # --------------------------------------------------------

        self._check_empty_dataset(
            train_data,
            "Training",
        )

        self._check_empty_dataset(
            test_data,
            "Testing",
        )

        if validation_data is not None:

            self._check_empty_dataset(
                validation_data,
                "Validation",
            )

        # --------------------------------------------------------
        # Column consistency
        # --------------------------------------------------------

        self._check_column_consistency(
            train_data,
            test_data,
            validation_data,
        )

        # --------------------------------------------------------
        # Missing values
        #
        # IMPORTANT:
        # Run this BEFORE target validation.
        # --------------------------------------------------------

        train_missing = (
            self._check_missing_values(
                train_data,
                "Training",
            )
        )

        test_missing = (
            self._check_missing_values(
                test_data,
                "Testing",
            )
        )

        validation_missing = {}

        if validation_data is not None:

            validation_missing = (
                self._check_missing_values(
                    validation_data,
                    "Validation",
                )
            )

        # --------------------------------------------------------
        # Target
        # --------------------------------------------------------

        target_column = (
            self._validate_target(
                train_data
            )
        )

        # --------------------------------------------------------
        # Duplicate rows
        # --------------------------------------------------------

        train_duplicates = (
            self._check_duplicates(
                train_data,
                "Training",
            )
        )

        test_duplicates = (
            self._check_duplicates(
                test_data,
                "Testing",
            )
        )

        validation_duplicates = 0

        if validation_data is not None:

            validation_duplicates = (
                self._check_duplicates(
                    validation_data,
                    "Validation",
                )
            )

        # --------------------------------------------------------
        # Feature analysis
        # --------------------------------------------------------

        features = (
            self._analyze_features(
                train_data,
                target_column,
            )
        )

        # --------------------------------------------------------
        # Statistics
        # --------------------------------------------------------

        statistics = {

            "train":
                self._dataset_statistics(
                    train_data
                ),

            "test":
                self._dataset_statistics(
                    test_data
                ),
        }

        if validation_data is not None:

            statistics[
                "validation"
            ] = self._dataset_statistics(
                validation_data
            )

        # --------------------------------------------------------
        # Build report
        # --------------------------------------------------------

        report = {

            "status": "passed",

            "target_column":
                target_column,

            "target_validation_deferred":
                target_column is None,

            "features":
                features,

            "missing_values": {

                "train":
                    train_missing,

                "validation":
                    validation_missing,

                "test":
                    test_missing,
            },

            "duplicates": {

                "train":
                    train_duplicates,

                "validation":
                    validation_duplicates,

                "test":
                    test_duplicates,
            },

            "statistics":
                statistics,
        }

        # --------------------------------------------------------
        # Save report
        # --------------------------------------------------------

        report_path = (
            self._save_report(
                report
            )
        )

        # --------------------------------------------------------
        # Final status
        # --------------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "DATA VALIDATION PASSED"
        )

        print(
            "=" * 60
        )

        print(
            f"\nValidation report saved:"
            f"\n{report_path}"
        )

        print(
            "\nData validation completed successfully."
        )

        return report