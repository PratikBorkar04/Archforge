"""
dataset_profiler.py

Generic dataset profiling module for ArchForge.

Responsibilities
----------------
1. Analyze dataset size.
2. Analyze feature/sample ratio.
3. Detect numerical and categorical features.
4. Detect regression vs classification.
5. Detect dataset complexity.
6. Provide model-selection metadata.
7. Determine whether automated training is safe.
8. Provide warnings for extremely small datasets.

Important
---------
The profiler DOES NOT modify the dataset.

It only analyzes the training data and produces metadata
for later stages such as:

    Model Selection
    Model Training
    Model Evaluation
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


# ============================================================
# Terminal Colors
# ============================================================

YELLOW = "\033[93m"
RESET = "\033[0m"


class DatasetProfiler:
    """
    Analyze a training dataset and generate a dataset profile.
    """

    # ============================================================
    # Dataset Size Thresholds
    # ============================================================

    VERY_SMALL_DATASET = 20
    SMALL_DATASET = 500
    MEDIUM_DATASET = 5000
    LARGE_DATASET = 50000

    # ============================================================
    # Training Safety Thresholds
    # ============================================================

    ABSOLUTE_MIN_ROWS = 20

    CAUTION_ROWS = 50

    MIN_SAMPLES_PER_FEATURE = 5

    HIGH_DIMENSIONAL_SAMPLES_PER_FEATURE = 10

    MIN_CLASSIFICATION_ROWS = 50

    MIN_SAMPLES_PER_CLASS = 5

    # ============================================================
    # Classification Detection Thresholds
    # ============================================================

    # Maximum number of unique numerical values that can
    # automatically be considered classification.
    #
    # Examples:
    #
    # 0 / 1
    # 1 / 2 / 3 / 4 / 5
    # Play / Not Play is handled separately as string data.
    #
    NUMERICAL_CLASSIFICATION_UNIQUE_THRESHOLD = 10

    # If the number of unique numerical values is a very
    # small fraction of the total number of samples, consider
    # the target classification.
    #
    # Example:
    #
    # 5 unique values / 1000 samples
    #
    # 5 / 1000 = 0.005
    #
    # -> CLASSIFICATION
    #
    NUMERICAL_CLASSIFICATION_RATIO_THRESHOLD = 0.05

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(
        self,
        project_root: Path,
        target_column: str,
    ) -> None:

        self.project_root = Path(
            project_root
        ).resolve()

        self.target_column = target_column

    # ============================================================
    # Main Profiling Method
    # ============================================================

    def profile(
        self,
        data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Profile the supplied training dataset.

        Parameters
        ----------
        data : pandas.DataFrame
            Training dataset.

        Returns
        -------
        Dict[str, Any]
            Dataset profile.
        """

        # --------------------------------------------------------
        # Validate dataset
        # --------------------------------------------------------

        if data is None:

            raise ValueError(
                "Dataset cannot be None."
            )

        if data.empty:

            raise ValueError(
                "Dataset is empty."
            )

        if (
            self.target_column
            not in data.columns
        ):

            raise ValueError(
                f"Target column "
                f"'{self.target_column}' "
                f"not found in dataset."
            )

        # ========================================================
        # Basic Information
        # ========================================================

        rows = len(data)

        feature_columns = [
            column
            for column in data.columns
            if column != self.target_column
        ]

        feature_count = len(
            feature_columns
        )

        # ========================================================
        # Feature Types
        # ========================================================

        feature_data = data[
            feature_columns
        ]

        numerical_features = (
            feature_data
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        categorical_features = (
            feature_data
            .select_dtypes(
                exclude=np.number
            )
            .columns
            .tolist()
        )

        numerical_count = len(
            numerical_features
        )

        categorical_count = len(
            categorical_features
        )

        # ========================================================
        # Target Information
        # ========================================================

        target = data[
            self.target_column
        ]

        target_clean = target.dropna()

        if target_clean.empty:

            raise ValueError(
                "Target column contains "
                "no valid values."
            )

        target_unique_values = int(
            target_clean.nunique()
        )

        target_dtype = str(
            target.dtype
        )

        # ========================================================
        # Problem Type
        # ========================================================

        problem_type = (
            self._detect_problem_type(
                target_clean
            )
        )

        # ========================================================
        # Dataset Size
        # ========================================================

        dataset_size = (
            self._detect_dataset_size(
                rows
            )
        )

        # ========================================================
        # Feature / Sample Ratio
        # ========================================================

        feature_sample_ratio = (
            feature_count / rows
            if rows > 0
            else 0.0
        )

        samples_per_feature = (
            rows / feature_count
            if feature_count > 0
            else float("inf")
        )

        dimensionality_level = (
            self._detect_dimensionality(
                feature_sample_ratio
            )
        )

        # ========================================================
        # Missing Values
        # ========================================================

        total_missing_values = int(
            data.isnull()
            .sum()
            .sum()
        )

        missing_percentage = (
            total_missing_values
            / data.size
            * 100
            if data.size > 0
            else 0.0
        )

        # ========================================================
        # Duplicate Rows
        # ========================================================

        duplicate_rows = int(
            data.duplicated()
            .sum()
        )

        duplicate_percentage = (
            duplicate_rows
            / rows
            * 100
            if rows > 0
            else 0.0
        )

        # ========================================================
        # Target Statistics
        # ========================================================

        class_distribution = {}

        min_class_samples = None

        if problem_type == "CLASSIFICATION":

            value_counts = (
                target_clean
                .value_counts()
            )

            class_distribution = {
                str(index): int(value)
                for index, value
                in value_counts.items()
            }

            if not value_counts.empty:

                min_class_samples = int(
                    value_counts.min()
                )

        # ========================================================
        # Dataset Complexity
        # ========================================================

        complexity = (
            self._detect_complexity(
                rows=rows,
                feature_count=feature_count,
                categorical_count=categorical_count,
                feature_sample_ratio=(
                    feature_sample_ratio
                ),
            )
        )

        # ========================================================
        # Training Safety
        # ========================================================

        safety = (
            self._training_safety_check(
                rows=rows,
                feature_count=feature_count,
                problem_type=problem_type,
                samples_per_feature=(
                    samples_per_feature
                ),
                min_class_samples=(
                    min_class_samples
                ),
            )
        )

        # ========================================================
        # Build Profile
        # ========================================================

        profile = {

            # ----------------------------------------------------
            # Dataset
            # ----------------------------------------------------

            "rows":
                rows,

            "features":
                feature_count,

            "numerical_features":
                numerical_features,

            "numerical_feature_count":
                numerical_count,

            "categorical_features":
                categorical_features,

            "categorical_feature_count":
                categorical_count,

            # ----------------------------------------------------
            # Target
            # ----------------------------------------------------

            "target_column":
                self.target_column,

            "target_dtype":
                target_dtype,

            "target_unique_values":
                target_unique_values,

            "problem_type":
                problem_type,

            "class_distribution":
                class_distribution,

            "minimum_class_samples":
                min_class_samples,

            # ----------------------------------------------------
            # Dataset size
            # ----------------------------------------------------

            "dataset_size":
                dataset_size,

            # ----------------------------------------------------
            # Dimensionality
            # ----------------------------------------------------

            "feature_sample_ratio":
                round(
                    feature_sample_ratio,
                    4,
                ),

            "samples_per_feature":
                round(
                    samples_per_feature,
                    4,
                )
                if np.isfinite(
                    samples_per_feature
                )
                else None,

            "dimensionality":
                dimensionality_level,

            # ----------------------------------------------------
            # Data quality
            # ----------------------------------------------------

            "missing_values":
                total_missing_values,

            "missing_percentage":
                round(
                    missing_percentage,
                    2,
                ),

            "duplicate_rows":
                duplicate_rows,

            "duplicate_percentage":
                round(
                    duplicate_percentage,
                    2,
                ),

            # ----------------------------------------------------
            # Complexity
            # ----------------------------------------------------

            "dataset_complexity":
                complexity,

            # ----------------------------------------------------
            # Training safety
            # ----------------------------------------------------

            "training_recommendation":
                safety["recommendation"],

            "training_safe":
                safety["safe"],

            "training_warnings":
                safety["warnings"],

            "training_reason":
                safety["reason"],
        }

        # ========================================================
        # Display
        # ========================================================

        self._display_profile(
            profile
        )

        return profile

    # ============================================================
    # Problem Type Detection
    # ============================================================

    @classmethod
    def _detect_problem_type(
        cls,
        target: pd.Series,
    ) -> str:
        """
        Detect classification vs regression.

        Rules
        -----

        String / object / category / bool
            -> CLASSIFICATION

        Numerical binary target
            -> CLASSIFICATION

        Numerical target with <= 10 unique values
            -> CLASSIFICATION

        Numerical target with a very small
        unique-value ratio
            -> CLASSIFICATION

        Otherwise numerical target
            -> REGRESSION

        Examples
        --------

        yes / no
            -> CLASSIFICATION

        play / not play
            -> CLASSIFICATION

        male / female
            -> CLASSIFICATION

        A / B / C
            -> CLASSIFICATION

        0 / 1
            -> CLASSIFICATION

        1 / 2 / 3 / 4 / 5
            -> CLASSIFICATION

        5 unique values / 1000 samples
            5 / 1000 = 0.005
            -> CLASSIFICATION

        500 unique values / 1000 samples
            500 / 1000 = 0.50
            -> REGRESSION

        Price:
            100000
            200000
            300000
            ...
            -> REGRESSION
        """

        target = target.dropna()

        if target.empty:

            raise ValueError(
                "Target column contains "
                "no valid values."
            )

        # ========================================================
        # Boolean
        # ========================================================

        if pd.api.types.is_bool_dtype(
            target
        ):

            return "CLASSIFICATION"

        # ========================================================
        # String / Object / Category
        # ========================================================

        # IMPORTANT:
        #
        # pd.api.types.is_object_dtype()
        # does not necessarily detect Arrow-backed
        # string columns.
        #
        # For example:
        #
        # dtype: str
        #
        # Therefore is_string_dtype() is also required.

        if (
            pd.api.types.is_object_dtype(
                target
            )
            or
            pd.api.types.is_string_dtype(
                target
            )
            or
            isinstance(
                target.dtype,
                pd.CategoricalDtype,
            )
        ):

            return "CLASSIFICATION"

        # ========================================================
        # Numerical
        # ========================================================

        if pd.api.types.is_numeric_dtype(
            target
        ):

            unique_values = (
                target.nunique()
            )

            total_samples = len(
                target
            )

            # ----------------------------------------------------
            # Unique-value ratio
            # ----------------------------------------------------

            unique_ratio = (
                unique_values
                / total_samples
                if total_samples > 0
                else 1.0
            )

            # ----------------------------------------------------
            # Binary target
            # ----------------------------------------------------

            if unique_values == 2:

                return "CLASSIFICATION"

            # ----------------------------------------------------
            # Small discrete numerical target
            # ----------------------------------------------------

            if (
                unique_values
                <= cls.NUMERICAL_CLASSIFICATION_UNIQUE_THRESHOLD
            ):

                return "CLASSIFICATION"

            # ----------------------------------------------------
            # Low unique-value ratio
            # ----------------------------------------------------

            if (
                unique_ratio
                <= cls.NUMERICAL_CLASSIFICATION_RATIO_THRESHOLD
            ):

                return "CLASSIFICATION"

            # ----------------------------------------------------
            # Continuous numerical target
            # ----------------------------------------------------

            return "REGRESSION"

        # ========================================================
        # Fallback
        # ========================================================

        return "REGRESSION"

    # ============================================================
    # Dataset Size
    # ============================================================

    @classmethod
    def _detect_dataset_size(
        cls,
        rows: int,
    ) -> str:
        """
        Categorize dataset based on row count.
        """

        if rows < cls.VERY_SMALL_DATASET:

            return "very_small"

        if rows < cls.SMALL_DATASET:

            return "small"

        if rows < cls.MEDIUM_DATASET:

            return "medium"

        if rows < cls.LARGE_DATASET:

            return "large"

        return "very_large"

    # ============================================================
    # Dimensionality
    # ============================================================

    @staticmethod
    def _detect_dimensionality(
        feature_sample_ratio: float,
    ) -> str:
        """
        Determine dimensionality relative to sample count.
        """

        if feature_sample_ratio >= 1.0:

            return "extreme"

        if feature_sample_ratio >= 0.5:

            return "high"

        if feature_sample_ratio >= 0.1:

            return "moderate"

        return "low"

    # ============================================================
    # Dataset Complexity
    # ============================================================

    @staticmethod
    def _detect_complexity(
        rows: int,
        feature_count: int,
        categorical_count: int,
        feature_sample_ratio: float,
    ) -> str:
        """
        Estimate dataset complexity.

        This is NOT a prediction of model performance.

        It is metadata used by the Model Selector.
        """

        # --------------------------------------------------------
        # Extremely small dataset
        # --------------------------------------------------------

        if rows < 20:

            return "EXTREME"

        # --------------------------------------------------------
        # High dimensionality
        # --------------------------------------------------------

        if feature_sample_ratio >= 0.5:

            return "HIGH"

        # --------------------------------------------------------
        # Many categorical features relative to data
        # --------------------------------------------------------

        if (
            categorical_count > 10
            and rows < 1000
        ):

            return "HIGH"

        # --------------------------------------------------------
        # Large dataset
        # --------------------------------------------------------

        if rows >= 5000:

            return "HIGH"

        # --------------------------------------------------------
        # Medium dataset
        # --------------------------------------------------------

        if rows >= 500:

            return "MODERATE"

        # --------------------------------------------------------
        # Small but usable dataset
        # --------------------------------------------------------

        return "LOW"

    # ============================================================
    # Training Safety
    # ============================================================

    @classmethod
    def _training_safety_check(
        cls,
        rows: int,
        feature_count: int,
        problem_type: str,
        samples_per_feature: float,
        min_class_samples,
    ) -> Dict[str, Any]:
        """
        Determine whether automated training is appropriate.

        IMPORTANT
        ---------
        This does not determine which model should be used.

        Model selection is handled by ModelSelector.

        The profiler only determines whether automated training
        is:

            SAFE
            CAUTION
            UNSAFE
        """

        warnings = []

        # ========================================================
        # 1. Absolute minimum
        # ========================================================

        if rows < cls.ABSOLUTE_MIN_ROWS:

            warnings.append(
                "Dataset contains fewer than "
                f"{cls.ABSOLUTE_MIN_ROWS} training rows."
            )

            return {

                "safe":
                    False,

                "recommendation":
                    "UNSAFE",

                "reason":
                    "Insufficient training samples "
                    "for reliable automated training.",

                "warnings":
                    warnings,
            }

        # ========================================================
        # 2. High dimensionality
        # ========================================================

        if (
            feature_count > 0
            and
            samples_per_feature
            < cls.MIN_SAMPLES_PER_FEATURE
        ):

            warnings.append(
                "There are too few training samples "
                "relative to the number of features."
            )

            return {

                "safe":
                    False,

                "recommendation":
                    "UNSAFE",

                "reason":
                    "Feature-to-sample ratio is too high.",

                "warnings":
                    warnings,
            }

        # ========================================================
        # 3. Classification checks
        # ========================================================

        if (
            problem_type
            == "CLASSIFICATION"
        ):

            if rows < cls.MIN_CLASSIFICATION_ROWS:

                warnings.append(
                    "Classification dataset contains "
                    f"fewer than "
                    f"{cls.MIN_CLASSIFICATION_ROWS} "
                    "samples."
                )

                return {

                    "safe":
                        False,

                    "recommendation":
                        "UNSAFE",

                    "reason":
                        "Too few samples for reliable "
                        "classification.",

                    "warnings":
                        warnings,
                }

            if (
                min_class_samples is not None
                and
                min_class_samples
                < cls.MIN_SAMPLES_PER_CLASS
            ):

                warnings.append(
                    "At least one class contains fewer "
                    f"than "
                    f"{cls.MIN_SAMPLES_PER_CLASS} "
                    "samples."
                )

                return {

                    "safe":
                        False,

                    "recommendation":
                        "UNSAFE",

                    "reason":
                        "One or more classes have "
                        "insufficient samples.",

                    "warnings":
                        warnings,
                }

        # ========================================================
        # 4. Small dataset caution
        # ========================================================

        if rows < cls.CAUTION_ROWS:

            warnings.append(
                "Dataset is small. Model performance "
                "may be unstable and should be interpreted "
                "with caution."
            )

            return {

                "safe":
                    True,

                "recommendation":
                    "CAUTION",

                "reason":
                    "Dataset is usable but small.",

                "warnings":
                    warnings,
            }

        # ========================================================
        # 5. High dimensionality warning
        # ========================================================

        if (
            feature_count > 0
            and
            samples_per_feature
            < cls.HIGH_DIMENSIONAL_SAMPLES_PER_FEATURE
        ):

            warnings.append(
                "Dataset has relatively few samples "
                "per feature. Model selection should "
                "prefer simpler models."
            )

            return {

                "safe":
                    True,

                "recommendation":
                    "CAUTION",

                "reason":
                    "Dataset has limited samples "
                    "relative to feature count.",

                "warnings":
                    warnings,
            }

        # ========================================================
        # 6. Normal dataset
        # ========================================================

        return {

            "safe":
                True,

            "recommendation":
                "SAFE",

            "reason":
                "Dataset contains sufficient samples "
                "for automated model selection.",

            "warnings":
                warnings,
        }

    # ============================================================
    # Display
    # ============================================================

    @staticmethod
    def _display_profile(
        profile: Dict[str, Any],
    ) -> None:
        """
        Display the dataset profile.

        Warnings are deliberately displayed at the END so the
        normal dataset information remains easy to read.
        """

        print(
            "\n"
            + "=" * 60
        )

        print(
            "DATASET PROFILING"
        )

        print(
            "=" * 60
        )

        # --------------------------------------------------------
        # Basic information
        # --------------------------------------------------------

        print(
            f"\nRows                 : "
            f"{profile['rows']}"
        )

        print(
            f"Features             : "
            f"{profile['features']}"
        )

        print(
            f"Numerical features   : "
            f"{profile['numerical_feature_count']}"
        )

        print(
            f"Categorical features : "
            f"{profile['categorical_feature_count']}"
        )

        # --------------------------------------------------------
        # Dataset size
        # --------------------------------------------------------

        print(
            f"Dataset size         : "
            f"{profile['dataset_size']}"
        )

        print(
            f"Feature/sample ratio : "
            f"{profile['feature_sample_ratio']}"
        )

        print(
            f"Samples/feature      : "
            f"{profile['samples_per_feature']}"
        )

        print(
            f"Feature/sample level : "
            f"{profile['dimensionality']}"
        )

        # --------------------------------------------------------
        # Problem
        # --------------------------------------------------------

        print(
            f"Problem type         : "
            f"{profile['problem_type']}"
        )

        print(
            f"Target unique values : "
            f"{profile['target_unique_values']}"
        )

        # --------------------------------------------------------
        # Data quality
        # --------------------------------------------------------

        print(
            f"Missing values       : "
            f"{profile['missing_values']}"
        )

        print(
            f"Duplicate rows       : "
            f"{profile['duplicate_rows']}"
        )

        # --------------------------------------------------------
        # Complexity
        # --------------------------------------------------------

        print(
            f"Dataset complexity   : "
            f"{profile['dataset_complexity']}"
        )

        # ========================================================
        # Training Recommendation
        # ========================================================

        print(
            "\n"
            + "=" * 60
        )

        print(
            "TRAINING RECOMMENDATION"
        )

        print(
            "=" * 60
        )

        print(
            f"\nTraining safety      : "
            f"{profile['training_recommendation']}"
        )

        print(
            f"Training allowed     : "
            f"{profile['training_safe']}"
        )

        print(
            f"Reason               : "
            f"{profile['training_reason']}"
        )

        # ========================================================
        # Warning at END
        # ========================================================

        warnings = profile[
            "training_warnings"
        ]

        if warnings:

            print(
                "\n"
                + "!" * 60
            )

            print(
                "WARNING"
            )

            print(
                "!" * 60
            )

            for warning in warnings:

                print(
                    f"\n⚠ {warning}"
                )

            if not profile[
                "training_safe"
            ]:

                print(
                    "\nAutomatic model training "
                    "is not recommended for this dataset."
                )

                print(
                    "You may still choose to proceed "
                    "manually if you understand the risks."
                )

            else:

                print(
                    "\nAutomated training can proceed, "
                    "but results should be interpreted "
                    "with caution."
                )

        print(
            "\nDataset profiling completed."
        )