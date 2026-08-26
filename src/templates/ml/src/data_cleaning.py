"""
data_cleaning.py

Automatic data cleaning module for ArchForge.

Responsibilities
----------------
1. Detect missing values.
2. Remove rows with missing target values.
3. Impute numerical missing values using median.
4. Impute categorical missing values using mode.
5. Remove columns with excessive missing values.
6. Remove constant columns.
7. Remove obvious ID/index columns.

The cleaning decisions are learned from the training dataset
and then applied consistently to validation and testing data.

This prevents data leakage.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


class DataCleaning:
    """
    Generic automatic data-cleaning component.

    The component learns cleaning rules from training data
    and applies the same rules to validation/test data.
    """

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    # If more than this percentage of a column is missing,
    # the column will be removed.
    MISSING_COLUMN_THRESHOLD = 0.50

    # Columns with only one unique value contain no useful
    # information for most ML models.
    CONSTANT_COLUMN_THRESHOLD = 1

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------

    def __init__(
        self,
        project_root: Path,
        target_column: str,
    ) -> None:

        self.project_root = Path(
            project_root
        ).resolve()

        self.target_column = target_column

        # Rules learned from training data
        self.columns_to_drop: List[str] = []

        self.numeric_fill_values: Dict[
            str,
            Any
        ] = {}

        self.categorical_fill_values: Dict[
            str,
            Any
        ] = {}

    # ============================================================
    # Main Method
    # ============================================================

    def initiate_data_cleaning(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
    ):
        """
        Clean training, validation and testing datasets.

        Cleaning rules are learned ONLY from training data.

        Returns
        -------
        tuple
            cleaned_train,
            cleaned_validation,
            cleaned_test,
            cleaning_report
        """

        print(
            "\n"
            + "=" * 60
        )

        print(
            "DATA CLEANING"
        )

        print(
            "=" * 60
        )

        if train_data is None:
            raise ValueError(
                "Training data cannot be None."
            )

        if train_data.empty:
            raise ValueError(
                "Training data is empty."
            )

        if (
            self.target_column
            not in train_data.columns
        ):
            raise ValueError(
                f"Target column "
                f"'{self.target_column}' "
                f"not found."
            )

        # --------------------------------------------------------
        # Work on copies
        # --------------------------------------------------------

        train_data = train_data.copy()

        if validation_data is not None:
            validation_data = (
                validation_data.copy()
            )

        if test_data is not None:
            test_data = test_data.copy()

        # --------------------------------------------------------
        # 1. Remove rows with missing target
        # --------------------------------------------------------

        train_before = len(train_data)

        train_data = train_data.dropna(
            subset=[
                self.target_column
            ]
        ).reset_index(drop=True)

        removed_target_rows = (
            train_before
            - len(train_data)
        )

        if removed_target_rows > 0:

            print(
                f"\nRemoved {removed_target_rows} "
                f"training rows with missing target."
            )

        # --------------------------------------------------------
        # Validation/test rows with missing target
        #
        # These rows cannot be evaluated, so remove them.
        # --------------------------------------------------------

        validation_target_rows_removed = 0

        if (
            validation_data is not None
            and self.target_column
            in validation_data.columns
        ):

            before = len(
                validation_data
            )

            validation_data = (
                validation_data
                .dropna(
                    subset=[
                        self.target_column
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            validation_target_rows_removed = (
                before
                - len(validation_data)
            )

        test_target_rows_removed = 0

        if (
            test_data is not None
            and self.target_column
            in test_data.columns
        ):

            before = len(
                test_data
            )

            test_data = (
                test_data
                .dropna(
                    subset=[
                        self.target_column
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            test_target_rows_removed = (
                before
                - len(test_data)
            )

        # --------------------------------------------------------
        # 2. Detect columns to drop
        #
        # IMPORTANT:
        # This is learned ONLY from training data.
        # --------------------------------------------------------

        self.columns_to_drop = (
            self._detect_columns_to_drop(
                train_data
            )
        )

        # Never drop the target.
        self.columns_to_drop = [
            column
            for column
            in self.columns_to_drop
            if column
            != self.target_column
        ]

        # --------------------------------------------------------
        # Apply column removal
        # --------------------------------------------------------

        if self.columns_to_drop:

            print(
                "\nColumns removed:"
            )

            for column in (
                self.columns_to_drop
            ):

                print(
                    f"  • {column}"
                )

            train_data = (
                train_data.drop(
                    columns=
                    self.columns_to_drop,
                    errors="ignore"
                )
            )

            if validation_data is not None:

                validation_data = (
                    validation_data.drop(
                        columns=
                        self.columns_to_drop,
                        errors="ignore"
                    )
                )

            if test_data is not None:

                test_data = (
                    test_data.drop(
                        columns=
                        self.columns_to_drop,
                        errors="ignore"
                    )
                )

        else:

            print(
                "\nNo columns required removal."
            )

        # --------------------------------------------------------
        # 3. Learn imputation values
        #
        # ONLY from training data.
        # --------------------------------------------------------

        self._learn_imputation_values(
            train_data
        )

        # --------------------------------------------------------
        # 4. Apply imputation
        # --------------------------------------------------------

        train_data = (
            self._apply_imputation(
                train_data
            )
        )

        if validation_data is not None:

            validation_data = (
                self._apply_imputation(
                    validation_data
                )
            )

        if test_data is not None:

            test_data = (
                self._apply_imputation(
                    test_data
                )
            )

        # --------------------------------------------------------
        # Cleaning report
        # --------------------------------------------------------

        report = {

            "target_column":
                self.target_column,

            "removed_columns":
                self.columns_to_drop,

            "numeric_imputation":
                self.numeric_fill_values,

            "categorical_imputation":
                self.categorical_fill_values,

            "removed_training_target_rows":
                removed_target_rows,

            "removed_validation_target_rows":
                validation_target_rows_removed,

            "removed_testing_target_rows":
                test_target_rows_removed,

            "training_rows_after_cleaning":
                len(train_data),

            "validation_rows_after_cleaning":
                (
                    len(validation_data)
                    if validation_data is not None
                    else None
                ),

            "testing_rows_after_cleaning":
                (
                    len(test_data)
                    if test_data is not None
                    else None
                ),
        }

        # --------------------------------------------------------
        # Display summary
        # --------------------------------------------------------

        self._display_summary(
            train_data,
            validation_data,
            test_data,
            report,
        )

        return (
            train_data,
            validation_data,
            test_data,
            report,
        )

    # ============================================================
    # Detect Columns To Drop
    # ============================================================

    def _detect_columns_to_drop(
        self,
        train_data: pd.DataFrame,
    ) -> List[str]:
        """
        Identify columns that are highly likely to be useless.

        Conservative rules are used here.

        A column is removed when:

        1. Missing percentage > threshold
        2. It is constant
        3. It looks like an obvious dataframe index
        """

        columns_to_drop = []

        for column in train_data.columns:

            if (
                column
                == self.target_column
            ):
                continue

            series = train_data[
                column
            ]

            # ----------------------------------------------------
            # Excessive missing values
            # ----------------------------------------------------

            missing_ratio = (
                series.isnull()
                .mean()
            )

            if (
                missing_ratio
                >
                self.MISSING_COLUMN_THRESHOLD
            ):

                columns_to_drop.append(
                    column
                )

                print(
                    f"\nDrop candidate: "
                    f"{column}"
                )

                print(
                    f"  Reason: "
                    f"{missing_ratio:.1%} "
                    f"missing values"
                )

                continue

            # ----------------------------------------------------
            # Constant column
            # ----------------------------------------------------

            unique_values = (
                series.nunique(
                    dropna=True
                )
            )

            if (
                unique_values
                <= self.CONSTANT_COLUMN_THRESHOLD
            ):

                columns_to_drop.append(
                    column
                )

                print(
                    f"\nDrop candidate: "
                    f"{column}"
                )

                print(
                    "  Reason: constant column"
                )

                continue

            # ----------------------------------------------------
            # Obvious index / ID columns
            # ----------------------------------------------------

            if self._is_index_like_column(
                column,
                series,
            ):

                columns_to_drop.append(
                    column
                )

                print(
                    f"\nDrop candidate: "
                    f"{column}"
                )

                print(
                    "  Reason: "
                    "identifier/index-like column"
                )

        return columns_to_drop

    # ============================================================
    # Detect ID / Index Columns
    # ============================================================

    @staticmethod
    def _is_index_like_column(
        column_name: str,
        series: pd.Series,
    ) -> bool:
        """
        Detect obvious index-like columns.

        Examples:

            id
            ID
            index
            row_id
            row_number
            unnamed: 0

        Also detects a numeric sequence such as:

            1, 2, 3, 4, 5, ...

        Conservative logic is used to avoid removing legitimate
        numerical features.
        """

        name = (
            str(column_name)
            .strip()
            .lower()
        )

        index_names = {

            "id",
            "index",
            "row_id",
            "rowid",
            "row_number",
            "row_number_id",
            "serial",
            "unnamed: 0",
        }

        if name in index_names:

            return True

        # --------------------------------------------------------
        # Check sequential numeric columns
        # --------------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            series
        ):

            values = (
                series
                .dropna()
                .to_numpy()
            )

            if len(values) > 1:

                expected = np.arange(
                    1,
                    len(values) + 1
                )

                if np.array_equal(
                    values,
                    expected
                ):

                    return True

                expected_zero = np.arange(
                    0,
                    len(values)
                )

                if np.array_equal(
                    values,
                    expected_zero
                ):

                    return True

        return False

    # ============================================================
    # Learn Imputation Values
    # ============================================================

    def _learn_imputation_values(
        self,
        train_data: pd.DataFrame,
    ) -> None:
        """
        Calculate missing-value replacement values.

        Numerical columns:
            Median

        Categorical columns:
            Mode

        These values are learned only from training data.
        """

        self.numeric_fill_values = {}

        self.categorical_fill_values = {}

        feature_columns = [
            column
            for column
            in train_data.columns
            if column
            != self.target_column
        ]

        for column in feature_columns:

            series = train_data[
                column
            ]

            # ----------------------------------------------------
            # Numerical
            # ----------------------------------------------------

            if pd.api.types.is_numeric_dtype(
                series
            ):

                if series.isnull().any():

                    median = (
                        series.median()
                    )

                    if pd.isna(
                        median
                    ):

                        median = 0

                    self.numeric_fill_values[
                        column
                    ] = median

            # ----------------------------------------------------
            # Categorical
            # ----------------------------------------------------

            else:

                if series.isnull().any():

                    mode = (
                        series.mode(
                            dropna=True
                        )
                    )

                    if not mode.empty:

                        fill_value = (
                            mode.iloc[0]
                        )

                    else:

                        fill_value = (
                            "Unknown"
                        )

                    self.categorical_fill_values[
                        column
                    ] = fill_value

    # ============================================================
    # Apply Imputation
    # ============================================================

    def _apply_imputation(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply previously learned imputation values.
        """

        data = data.copy()

        # --------------------------------------------------------
        # Numerical columns
        # --------------------------------------------------------

        for (
            column,
            value,
        ) in self.numeric_fill_values.items():

            if column in data.columns:

                data[column] = (
                    data[column]
                    .fillna(value)
                )

        # --------------------------------------------------------
        # Categorical columns
        # --------------------------------------------------------

        for (
            column,
            value,
        ) in (
            self.categorical_fill_values
            .items()
        ):

            if column in data.columns:

                data[column] = (
                    data[column]
                    .fillna(value)
                )

        return data

    # ============================================================
    # Display Summary
    # ============================================================

    @staticmethod
    def _display_summary(
        train_data: pd.DataFrame,
        validation_data: Optional[
            pd.DataFrame
        ],
        test_data: Optional[
            pd.DataFrame
        ],
        report: Dict[str, Any],
    ) -> None:
        """
        Display cleaning summary.
        """

        print(
            "\n"
            + "=" * 60
        )

        print(
            "DATA CLEANING SUMMARY"
        )

        print(
            "=" * 60
        )

        print(
            f"\nColumns removed : "
            f"{len(report['removed_columns'])}"
        )

        if report[
            "removed_columns"
        ]:

            for column in report[
                "removed_columns"
            ]:

                print(
                    f"  • {column}"
                )

        print(
            f"\nNumerical columns imputed : "
            f"{len(report['numeric_imputation'])}"
        )

        for (
            column,
            value,
        ) in report[
            "numeric_imputation"
        ].items():

            print(
                f"  • {column} → "
                f"median ({value})"
            )

        print(
            f"\nCategorical columns imputed : "
            f"{len(report['categorical_imputation'])}"
        )

        for (
            column,
            value,
        ) in report[
            "categorical_imputation"
        ].items():

            print(
                f"  • {column} → "
                f"mode ({value})"
            )

        print(
            f"\nTraining rows : "
            f"{len(train_data)}"
        )

        print(
            f"Validation rows : "
            f"{len(validation_data)}"
            if validation_data is not None
            else
            "Validation rows : Not provided"
        )

        print(
            f"Testing rows : "
            f"{len(test_data)}"
            if test_data is not None
            else
            "Testing rows : Not provided"
        )

        print(
            "\nData cleaning completed successfully."
        )