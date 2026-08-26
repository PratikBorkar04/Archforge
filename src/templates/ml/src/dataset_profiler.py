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
7. Warn when the dataset is too small for reliable training.

Important
---------
The profiler does NOT modify the dataset.
It only analyzes the training data and produces metadata.
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


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

        numerical_features = (
            data[
                feature_columns
            ]
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        categorical_features = (
            data[
                feature_columns
            ]
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

        target_unique_values = (
            target.nunique(
                dropna=True
            )
        )

        target_dtype = str(
            target.dtype
        )

        # ========================================================
        # Problem Type
        # ========================================================

        problem_type = (
            self._detect_problem_type(
                target
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

        if rows > 0:

            feature_sample_ratio = (
                feature_count
                / rows
            )

        else:

            feature_sample_ratio = 0.0

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
            else 0
        )

        # ========================================================
        # Duplicate Rows
        # ========================================================

        duplicate_rows = int(
            data.duplicated()
            .sum()
        )

        # ========================================================
        # Dataset Complexity
        # ========================================================

        complexity = (
            self._detect_complexity(
                rows=rows,
                feature_count=feature_count,
                categorical_count=
                    categorical_count,
                feature_sample_ratio=
                    feature_sample_ratio,
            )
        )

        # ========================================================
        # Training Safety
        # ========================================================

        training_recommendation = (
            self._training_safety_check(
                rows=rows,
                feature_count=feature_count,
                problem_type=problem_type,
            )
        )

        # ========================================================
        # Build Profile
        # ========================================================

        profile = {

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

            "target_column":
                self.target_column,

            "target_dtype":
                target_dtype,

            "target_unique_values":
                int(
                    target_unique_values
                ),

            "problem_type":
                problem_type,

            "dataset_size":
                dataset_size,

            "feature_sample_ratio":
                round(
                    feature_sample_ratio,
                    4,
                ),

            "dimensionality":
                dimensionality_level,

            "missing_values":
                total_missing_values,

            "missing_percentage":
                round(
                    missing_percentage,
                    2,
                ),

            "duplicate_rows":
                duplicate_rows,

            "dataset_complexity":
                complexity,

            "training_recommendation":
                training_recommendation,

            "training_safe":
                training_recommendation
                == "SAFE",
        }

        # ========================================================
        # Display Profile
        # ========================================================

        self._display_profile(
            profile
        )

        return profile

    # ============================================================
    # Problem Type Detection
    # ============================================================

    @staticmethod
    def _detect_problem_type(
        target: pd.Series,
    ) -> str:
        """
        Detect whether the ML problem is classification
        or regression.

        Rules
        -----
        1. Object/string/category/bool target
           -> CLASSIFICATION

        2. Numerical target:
           - continuous values
           -> REGRESSION

           - binary/discrete class-like values
           -> CLASSIFICATION

        Important:
        A numerical target with only a few values is NOT
        automatically considered classification.

        Example:

            Price:
            100000
            200000
            300000

        is still REGRESSION.
        """

        # --------------------------------------------------------
        # Remove missing values for analysis
        # --------------------------------------------------------

        target = target.dropna()

        if target.empty:

            raise ValueError(
                "Target column contains "
                "no valid values."
            )

        # --------------------------------------------------------
        # Boolean
        # --------------------------------------------------------

        if pd.api.types.is_bool_dtype(
            target
        ):

            return "CLASSIFICATION"

        # --------------------------------------------------------
        # String / Object / Category
        # --------------------------------------------------------

        if (
            pd.api.types.is_object_dtype(
                target
            )
            or
            pd.api.types.is_categorical_dtype(
                target
            )
        ):

            return "CLASSIFICATION"

        # --------------------------------------------------------
        # Numerical target
        # --------------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            target
        ):

            unique_values = (
                target.nunique()
            )

            total_values = len(
                target
            )

            # ----------------------------------------------------
            # Binary numerical target
            #
            # Examples:
            # 0 / 1
            # True / False represented numerically
            # ----------------------------------------------------

            if unique_values == 2:

                return "CLASSIFICATION"

            # ----------------------------------------------------
            # Small discrete integer targets
            #
            # Examples:
            #
            # Rating:
            # 1,2,3,4,5
            #
            # Class:
            # 0,1,2
            #
            # ----------------------------------------------------

            if pd.api.types.is_integer_dtype(
                target
            ):

                unique_ratio = (
                    unique_values
                    / total_values
                )

                # Small number of unique integer
                # values compared with dataset size.
                #
                # Example:
                #
                # 1000 rows
                # 3 classes
                #
                # -> Classification
                #
                # 1000 rows
                # 850 unique ages
                #
                # -> Regression
                #

                if (
                    unique_values <= 20
                    and unique_ratio <= 0.05
                ):

                    return "CLASSIFICATION"

            # ----------------------------------------------------
            # Otherwise numerical target is continuous.
            # ----------------------------------------------------

            return "REGRESSION"

        # --------------------------------------------------------
        # Fallback
        # --------------------------------------------------------

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
        Categorize dataset based on number of rows.
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
        Determine feature/sample dimensionality.

        Lower ratio:
            More samples per feature.

        Higher ratio:
            More features relative to samples.
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

        This is NOT a model-performance prediction.

        It is only used to help narrow down candidate models.
        """

        # --------------------------------------------------------
        # Extremely small data
        # --------------------------------------------------------

        if rows < 20:

            return "EXTREME"

        # --------------------------------------------------------
        # High-dimensional relative to samples
        # --------------------------------------------------------

        if feature_sample_ratio >= 0.5:

            return "HIGH"

        # --------------------------------------------------------
        # Many categorical features
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

        return "LOW"

    # ============================================================
    # Training Safety
    # ============================================================

    @staticmethod
    def _training_safety_check(
        rows: int,
        feature_count: int,
        problem_type: str,
    ) -> str:
        """
        Determine whether the dataset contains enough samples
        for meaningful automated model training.

        This is deliberately conservative.

        The model selector can later use more sophisticated
        rules based on the selected algorithm.
        """

        # --------------------------------------------------------
        # Absolute minimum
        # --------------------------------------------------------

        if rows < 20:

            return "UNSAFE"

        # --------------------------------------------------------
        # Too many features for available samples
        # --------------------------------------------------------

        if (
            feature_count > 0
            and rows
            < feature_count * 5
        ):

            return "UNSAFE"

        # --------------------------------------------------------
        # Classification needs enough examples per class
        # --------------------------------------------------------
        #
        # Detailed class-balance checks can be added later.
        # --------------------------------------------------------

        if (
            problem_type
            == "CLASSIFICATION"
            and rows < 50
        ):

            return "UNSAFE"

        return "SAFE"

    # ============================================================
    # Display
    # ============================================================

    @staticmethod
    def _display_profile(
        profile: Dict[str, Any],
    ) -> None:
        """
        Print dataset profile to terminal.
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

        print(
            f"Dataset size         : "
            f"{profile['dataset_size']}"
        )

        print(
            f"Feature/sample ratio : "
            f"{profile['feature_sample_ratio']}"
        )

        print(
            f"Feature/sample level : "
            f"{profile['dimensionality']}"
        )

        print(
            f"Problem type         : "
            f"{profile['problem_type']}"
        )

        print(
            f"Target unique values : "
            f"{profile['target_unique_values']}"
        )

        print(
            f"Missing values       : "
            f"{profile['missing_values']}"
        )

        print(
            f"Duplicate rows       : "
            f"{profile['duplicate_rows']}"
        )

        print(
            f"Dataset complexity   : "
            f"{profile['dataset_complexity']}"
        )

        # --------------------------------------------------------
        # Training safety
        # --------------------------------------------------------

        print(
            f"\nTraining safety      : "
            f"{profile['training_recommendation']}"
        )

        if not profile[
            "training_safe"
        ]:

            print(
                "\nWARNING:"
            )

            print(
                "Dataset is too small or "
                "too complex for reliable "
                "automatic model training."
            )

        print(
            "\nDataset profiling completed."
        )