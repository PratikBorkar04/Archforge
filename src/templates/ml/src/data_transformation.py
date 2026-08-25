"""
data_transformation.py

Generic data transformation module for ArchForge.

Responsibilities
----------------
1. Separate features (X) and target (y).
2. Automatically detect numerical and categorical columns.
3. Handle missing numerical values.
4. Handle missing categorical values.
5. Scale numerical features.
6. One-hot encode nominal categorical features.
7. Support ordinal encoding when explicitly configured.
8. Fit preprocessing ONLY on training data.
9. Apply the same preprocessing to validation/test data.
10. Save the fitted preprocessor.

The generated ML project should require minimal
manual preprocessing from the user.
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
)


class DataTransformation:
    """
    Generic preprocessing pipeline for ML datasets.
    """

    def __init__(
        self,
        project_root: Path,
        target_column: Optional[str] = None,
        ordinal_columns: Optional[
            Dict[str, List[str]]
        ] = None,
    ) -> None:

        self.project_root = Path(
            project_root
        ).resolve()

        self.artifacts_dir = (
            self.project_root
            / "artifacts"
        )

        self.preprocessor_path = (
            self.artifacts_dir
            / "preprocessor.pkl"
        )

        # --------------------------------------------------------
        # Optional ordinal configuration
        #
        # Example:
        #
        # {
        #     "Quality": [
        #         "Low",
        #         "Medium",
        #         "High"
        #     ]
        # }
        #
        # If this is None, categorical features
        # are treated as nominal and One-Hot encoded.
        # --------------------------------------------------------

        self.ordinal_columns = (
            ordinal_columns or {}
        )

        self.target_column = target_column

        self.preprocessor = None

    # ============================================================
    # Target Detection
    # ============================================================

    @staticmethod
    def _detect_target_column(
        data: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> str:
        """
        Detect target column.

        Current ArchForge MVP:
        If target_column is not supplied,
        the last column is treated as target.
        """

        if target_column:

            if (
                target_column
                not in data.columns
            ):

                raise ValueError(
                    f"Target column "
                    f"'{target_column}' "
                    "was not found."
                )

            return target_column

        if data.empty:

            raise ValueError(
                "Cannot detect target from "
                "an empty dataset."
            )

        return str(
            data.columns[-1]
        )

    # ============================================================
    # Column Detection
    # ============================================================

    def _detect_feature_columns(
        self,
        data: pd.DataFrame,
        target_column: str,
    ) -> Tuple[
        List[str],
        List[str],
        List[str],
    ]:
        """
        Automatically detect:

        1. Numerical columns
        2. Nominal categorical columns
        3. Ordinal categorical columns
        """

        features = data.drop(
            columns=[target_column]
        )

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

        # --------------------------------------------------------
        # Find explicitly configured ordinal columns
        # --------------------------------------------------------

        ordinal_column_names = list(
            self.ordinal_columns.keys()
        )

        ordinal_columns = [
            column
            for column in ordinal_column_names
            if column in categorical_columns
        ]

        # --------------------------------------------------------
        # Remaining categorical columns
        # become nominal categorical columns.
        # --------------------------------------------------------

        nominal_columns = [
            column
            for column in categorical_columns
            if column not in ordinal_columns
        ]

        return (
            numerical_columns,
            nominal_columns,
            ordinal_columns,
        )

    # ============================================================
    # Build Preprocessor
    # ============================================================

    def _build_preprocessor(
        self,
        numerical_columns: List[str],
        nominal_columns: List[str],
        ordinal_columns: List[str],
    ) -> ColumnTransformer:
        """
        Build a generic preprocessing pipeline.
        """

        transformers = []

        # --------------------------------------------------------
        # Numerical pipeline
        # --------------------------------------------------------

        if numerical_columns:

            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]
            )

            transformers.append(
                (
                    "numerical",
                    numerical_pipeline,
                    numerical_columns,
                )
            )

        # --------------------------------------------------------
        # Nominal categorical pipeline
        # --------------------------------------------------------

        if nominal_columns:

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            )

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    nominal_columns,
                )
            )

        # --------------------------------------------------------
        # Ordinal categorical pipeline
        # --------------------------------------------------------

        if ordinal_columns:

            categories = [
                self.ordinal_columns[column]
                for column in ordinal_columns
            ]

            ordinal_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "ordinal_encoder",
                        OrdinalEncoder(
                            categories=categories,
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )

            transformers.append(
                (
                    "ordinal",
                    ordinal_pipeline,
                    ordinal_columns,
                )
            )

        if not transformers:

            raise ValueError(
                "No usable feature columns "
                "were detected."
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

    # ============================================================
    # Transform Data
    # ============================================================

    def _transform(
        self,
        data: pd.DataFrame,
        target_column: str,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate X and y and transform X.
        """

        X = data.drop(
            columns=[target_column]
        )

        y = data[target_column]

        transformed = (
            self.preprocessor.transform(X)
        )

        feature_names = (
            self.preprocessor
            .get_feature_names_out()
        )

        transformed_data = pd.DataFrame(
            transformed,
            columns=feature_names,
            index=data.index,
        )

        return (
            transformed_data,
            y.reset_index(drop=True),
        )

    # ============================================================
    # Save Preprocessor
    # ============================================================

    def _save_preprocessor(self) -> Path:
        """
        Save the fitted preprocessing pipeline.
        """

        self.artifacts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.preprocessor,
            self.preprocessor_path,
        )

        return self.preprocessor_path

    # ============================================================
    # Main Transformation
    # ============================================================

    def initiate_data_transformation(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        validation_data: Optional[
            pd.DataFrame
        ] = None,
    ) -> Dict[str, object]:
        """
        Execute complete data transformation.

        IMPORTANT
        ---------
        The preprocessor is fitted ONLY on
        training data.

        Validation and test datasets are
        transformed using the already fitted
        preprocessor.
        """

        print(
            "\n[3/6] Data Transformation"
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "DATA TRANSFORMATION"
        )

        print(
            "=" * 60
        )

        # --------------------------------------------------------
        # Target
        # --------------------------------------------------------

        target_column = (
            self._detect_target_column(
                train_data,
                self.target_column,
            )
        )

        print(
            f"\nTarget column: "
            f"{target_column}"
        )

        # --------------------------------------------------------
        # Detect feature types
        # --------------------------------------------------------

        (
            numerical_columns,
            nominal_columns,
            ordinal_columns,
        ) = self._detect_feature_columns(
            train_data,
            target_column,
        )

        print(
            "\nDetected feature types:"
        )

        print(
            f"  Numerical : "
            f"{len(numerical_columns)}"
        )

        print(
            f"  Categorical : "
            f"{len(nominal_columns)}"
        )

        print(
            f"  Ordinal : "
            f"{len(ordinal_columns)}"
        )

        # --------------------------------------------------------
        # Display column names
        # --------------------------------------------------------

        if numerical_columns:

            print(
                "\nNumerical columns:"
            )

            for column in (
                numerical_columns
            ):

                print(
                    f"  • {column}"
                )

        if nominal_columns:

            print(
                "\nCategorical columns "
                "(One-Hot Encoding):"
            )

            for column in (
                nominal_columns
            ):

                print(
                    f"  • {column}"
                )

        if ordinal_columns:

            print(
                "\nOrdinal columns:"
            )

            for column in (
                ordinal_columns
            ):

                print(
                    f"  • {column}"
                )

        # --------------------------------------------------------
        # Build preprocessor
        # --------------------------------------------------------

        self.preprocessor = (
            self._build_preprocessor(
                numerical_columns,
                nominal_columns,
                ordinal_columns,
            )
        )

        # --------------------------------------------------------
        # FIT ONLY ON TRAINING DATA
        # --------------------------------------------------------

        X_train = train_data.drop(
            columns=[target_column]
        )

        y_train = train_data[
            target_column
        ]

        print(
            "\nFitting preprocessor "
            "using training data..."
        )

        self.preprocessor.fit(
            X_train
        )

        # --------------------------------------------------------
        # Transform training data
        # --------------------------------------------------------

        train_transformed = (
            self.preprocessor.transform(
                X_train
            )
        )

        feature_names = (
            self.preprocessor
            .get_feature_names_out()
        )

        X_train_transformed = (
            pd.DataFrame(
                train_transformed,
                columns=feature_names,
            )
        )

        y_train_transformed = (
            y_train.reset_index(
                drop=True
            )
        )

        # --------------------------------------------------------
        # Transform test data
        # --------------------------------------------------------

        X_test = test_data.drop(
            columns=[target_column]
        )

        y_test = test_data[
            target_column
        ]

        X_test_transformed = (
            pd.DataFrame(
                self.preprocessor.transform(
                    X_test
                ),
                columns=feature_names,
            )
        )

        y_test_transformed = (
            y_test.reset_index(
                drop=True
            )
        )

        # --------------------------------------------------------
        # Transform validation data
        # --------------------------------------------------------

        X_validation_transformed = (
            None
        )

        y_validation_transformed = (
            None
        )

        if validation_data is not None:

            X_validation = (
                validation_data.drop(
                    columns=[
                        target_column
                    ]
                )
            )

            y_validation = (
                validation_data[
                    target_column
                ]
            )

            X_validation_transformed = (
                pd.DataFrame(
                    self.preprocessor.transform(
                        X_validation
                    ),
                    columns=feature_names,
                )
            )

            y_validation_transformed = (
                y_validation.reset_index(
                    drop=True
                )
            )

        # --------------------------------------------------------
        # Save preprocessor
        # --------------------------------------------------------

        preprocessor_path = (
            self._save_preprocessor()
        )

        # --------------------------------------------------------
        # Summary
        # --------------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "TRANSFORMATION SUMMARY"
        )

        print(
            "=" * 60
        )

        print(
            f"\nOriginal features : "
            f"{len(X_train.columns)}"
        )

        print(
            f"Transformed features : "
            f"{len(feature_names)}"
        )

        print(
            f"Training rows : "
            f"{len(X_train_transformed)}"
        )

        if (
            X_validation_transformed
            is not None
        ):

            print(
                f"Validation rows : "
                f"{len(X_validation_transformed)}"
            )

        print(
            f"Testing rows : "
            f"{len(X_test_transformed)}"
        )

        print(
            f"\nPreprocessor saved:"
            f"\n{preprocessor_path}"
        )

        print(
            "\nData transformation "
            "completed successfully."
        )

        # --------------------------------------------------------
        # Return everything required
        # by model training.
        # --------------------------------------------------------

        return {
            "X_train":
                X_train_transformed,

            "y_train":
                y_train_transformed,

            "X_validation":
                X_validation_transformed,

            "y_validation":
                y_validation_transformed,

            "X_test":
                X_test_transformed,

            "y_test":
                y_test_transformed,

            "target_column":
                target_column,

            "feature_names":
                list(feature_names),

            "preprocessor":
                self.preprocessor,

            "preprocessor_path":
                preprocessor_path,
        }