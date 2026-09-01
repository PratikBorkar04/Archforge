"""
prediction_pipeline.py

Prediction component for ArchForge.

Responsibilities
----------------
1. Load the trained model.
2. Load the fitted preprocessor.
3. Load model metadata.
4. Discover the original input feature schema.
5. Identify numerical and categorical features.
6. Provide categorical values for UI generation.
7. Validate user input.
8. Apply the same preprocessing used during training.
9. Generate predictions.
10. Provide prediction information to the generic UI.

This module does NOT:
- Train models.
- Evaluate models.
- Select models.
- Save models.
- Handle UI.
"""

from pathlib import Path
from typing import Dict, Any, List

import json
import pickle

import pandas as pd


class PredictionPipeline:
    """
    Loads trained artifacts and generates predictions
    from raw user input.
    """

    # ============================================================
    # Initialization
    # ============================================================

    def __init__(
        self,
        project_root: Path,
    ) -> None:

        self.project_root = Path(
            project_root
        ).resolve()

        # --------------------------------------------------------
        # Model artifact
        # --------------------------------------------------------

        self.model_path = (
            self.project_root
            / "artifacts"
            / "models"
            / "best_model.pkl"
        )

        # --------------------------------------------------------
        # Preprocessor artifact
        # --------------------------------------------------------

        self.preprocessor_path = (
            self.project_root
            / "artifacts"
            / "preprocessor.pkl"
        )

        # --------------------------------------------------------
        # Model metadata
        # --------------------------------------------------------

        self.metadata_path = (
            self.project_root
            / "artifacts"
            / "models"
            / "model_metadata.json"
        )

        # --------------------------------------------------------
        # Loaded artifacts
        # --------------------------------------------------------

        self.model = None

        self.preprocessor = None

        self.metadata = {}

        self.feature_schema = []

    # ============================================================
    # Load Artifacts
    # ============================================================

    def load_artifacts(
        self,
    ) -> None:
        """
        Load model, preprocessor and metadata.
        """

        print(
            "\n" + "=" * 60
        )

        print(
            "LOADING PREDICTION ARTIFACTS"
        )

        print(
            "=" * 60
        )

        # --------------------------------------------------------
        # Validate model
        # --------------------------------------------------------

        if not self.model_path.exists():

            raise FileNotFoundError(
                "Trained model artifact was not found:\n"
                f"{self.model_path}"
            )

        # --------------------------------------------------------
        # Validate preprocessor
        # --------------------------------------------------------

        if not self.preprocessor_path.exists():

            raise FileNotFoundError(
                "Preprocessor artifact was not found:\n"
                f"{self.preprocessor_path}"
            )

        # --------------------------------------------------------
        # Load model
        # --------------------------------------------------------

        print(
            "\nLoading trained model..."
        )

        with open(
            self.model_path,
            "rb",
        ) as file:

            self.model = pickle.load(
                file
            )

        print(
            "✓ Model loaded."
        )

        # --------------------------------------------------------
        # Load preprocessor
        # --------------------------------------------------------

        print(
            "Loading preprocessor..."
        )

        with open(
            self.preprocessor_path,
            "rb",
        ) as file:

            self.preprocessor = pickle.load(
                file
            )

        print(
            "✓ Preprocessor loaded."
        )

        # --------------------------------------------------------
        # Load metadata
        # --------------------------------------------------------

        if self.metadata_path.exists():

            print(
                "Loading model metadata..."
            )

            with open(
                self.metadata_path,
                "r",
                encoding="utf-8",
            ) as file:

                self.metadata = json.load(
                    file
                )

            print(
                "✓ Model metadata loaded."
            )

        else:

            print(
                "⚠ Model metadata not found."
            )

            self.metadata = {}

        # --------------------------------------------------------
        # Build feature schema
        # --------------------------------------------------------

        self.feature_schema = (
            self._build_feature_schema()
        )

        # --------------------------------------------------------
        # Display loaded information
        # --------------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "PREDICTION ARTIFACTS"
        )

        print(
            "=" * 60
        )

        print(
            f"Model          : "
            f"{self.get_model_name()}"
        )

        print(
            f"Problem type   : "
            f"{self.get_problem_type()}"
        )

        print(
            f"Target column  : "
            f"{self.get_target_column()}"
        )

        print(
            f"Input features : "
            f"{len(self.feature_schema)}"
        )

        print(
            "\nFeatures:"
        )

        for index, feature in enumerate(
            self.feature_schema,
            start=1,
        ):

            print(
                f"  {index}. "
                f"{feature['name']} "
                f"({feature['type']})"
            )

        print(
            "\nPrediction artifacts "
            "loaded successfully."
        )

    # ============================================================
    # Build Feature Schema
    # ============================================================

    def _build_feature_schema(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Build the original input feature schema.

        Metadata is preferred because ModelPusher explicitly
        stores the original training feature names.

        The fitted preprocessor is used to determine
        numerical/categorical types and categories.
        """

        # --------------------------------------------------------
        # Metadata feature list
        # --------------------------------------------------------

        metadata_features = (
            self.metadata.get(
                "features"
            )
        )

        # --------------------------------------------------------
        # If metadata contains features,
        # use metadata as the source of truth.
        # --------------------------------------------------------

        if (
            metadata_features
            and isinstance(
                metadata_features,
                list,
            )
        ):

            schema = []

            for feature_name in (
                metadata_features
            ):

                feature_name = str(
                    feature_name
                )

                feature_info = (
                    self._find_feature_in_preprocessor(
                        feature_name
                    )
                )

                if feature_info:

                    schema.append(
                        feature_info
                    )

                else:

                    # ------------------------------------------------
                    # Fallback if preprocessor cannot identify
                    # the feature.
                    # ------------------------------------------------

                    schema.append({

                        "name":
                            feature_name,

                        "type":
                            "numerical",

                        "input_type":
                            "number",

                        "categories":
                            [],

                        "transformer":
                            "unknown",

                    })

            return schema

        # --------------------------------------------------------
        # Legacy fallback:
        # Discover features directly from preprocessor.
        # --------------------------------------------------------

        return (
            self._build_schema_from_preprocessor()
        )

    # ============================================================
    # Find Feature In Preprocessor
    # ============================================================

    def _find_feature_in_preprocessor(
        self,
        feature_name: str,
    ) -> Dict[str, Any]:
        """
        Find one original feature inside the fitted
        ColumnTransformer.
        """

        transformers = getattr(
            self.preprocessor,
            "transformers_",
            None,
        )

        if transformers is None:

            return {}

        for (
            transformer_name,
            transformer,
            columns,
        ) in transformers:

            # ----------------------------------------------------
            # Ignore dropped columns
            # ----------------------------------------------------

            if transformer == "drop":

                continue

            if columns is None:

                continue

            # ----------------------------------------------------
            # Normalize columns
            # ----------------------------------------------------

            if isinstance(
                columns,
                str,
            ):

                columns = [
                    columns
                ]

            else:

                try:

                    columns = list(
                        columns
                    )

                except TypeError:

                    continue

            if feature_name not in columns:

                continue

            # ----------------------------------------------------
            # Numerical transformer
            # ----------------------------------------------------

            if transformer_name in {

                "num",
                "numerical",
                "numeric",
                "numerical_pipeline",

            }:

                return {

                    "name":
                        feature_name,

                    "type":
                        "numerical",

                    "input_type":
                        "number",

                    "categories":
                        [],

                    "transformer":
                        transformer_name,

                }

            # ----------------------------------------------------
            # Categorical transformer
            # ----------------------------------------------------

            if transformer_name in {

                "cat",
                "categorical",
                "categorical_pipeline",

            }:

                categories = (
                    self._get_categories_for_feature(
                        transformer,
                        columns,
                        feature_name,
                    )
                )

                return {

                    "name":
                        feature_name,

                    "type":
                        "categorical",

                    "input_type":
                        "select",

                    "categories":
                        categories,

                    "transformer":
                        transformer_name,

                }

            # ----------------------------------------------------
            # Unknown transformer
            # ----------------------------------------------------

            inferred_type = (
                self._infer_transformer_type(
                    transformer
                )
            )

            categories = []

            if (
                inferred_type
                == "categorical"
            ):

                categories = (
                    self._get_categories_for_feature(
                        transformer,
                        columns,
                        feature_name,
                    )
                )

            return {

                "name":
                    feature_name,

                "type":
                    inferred_type,

                "input_type":
                    (
                        "select"
                        if inferred_type
                        == "categorical"
                        else "number"
                    ),

                "categories":
                    categories,

                "transformer":
                    transformer_name,

            }

        return {}

    # ============================================================
    # Build Schema From Preprocessor
    # ============================================================

    def _build_schema_from_preprocessor(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Legacy fallback that discovers the complete
        feature schema from the fitted preprocessor.
        """

        if self.preprocessor is None:

            raise RuntimeError(
                "Preprocessor must be loaded "
                "before building feature schema."
            )

        transformers = getattr(
            self.preprocessor,
            "transformers_",
            None,
        )

        if transformers is None:

            raise RuntimeError(
                "Loaded preprocessor does not contain "
                "fitted transformer information."
            )

        schema = []

        for (
            transformer_name,
            transformer,
            columns,
        ) in transformers:

            if transformer == "drop":

                continue

            if columns is None:

                continue

            if isinstance(
                columns,
                str,
            ):

                columns = [
                    columns
                ]

            else:

                try:

                    columns = list(
                        columns
                    )

                except TypeError:

                    continue

            # ----------------------------------------------------
            # Numerical
            # ----------------------------------------------------

            if transformer_name in {

                "num",
                "numerical",
                "numeric",
                "numerical_pipeline",

            }:

                for column in columns:

                    schema.append({

                        "name":
                            str(column),

                        "type":
                            "numerical",

                        "input_type":
                            "number",

                        "categories":
                            [],

                        "transformer":
                            transformer_name,

                    })

            # ----------------------------------------------------
            # Categorical
            # ----------------------------------------------------

            elif transformer_name in {

                "cat",
                "categorical",
                "categorical_pipeline",

            }:

                for column in columns:

                    categories = (
                        self._get_categories_for_feature(
                            transformer,
                            columns,
                            column,
                        )
                    )

                    schema.append({

                        "name":
                            str(column),

                        "type":
                            "categorical",

                        "input_type":
                            "select",

                        "categories":
                            categories,

                        "transformer":
                            transformer_name,

                    })

            # ----------------------------------------------------
            # Unknown
            # ----------------------------------------------------

            else:

                inferred_type = (
                    self._infer_transformer_type(
                        transformer
                    )
                )

                for column in columns:

                    categories = []

                    if (
                        inferred_type
                        == "categorical"
                    ):

                        categories = (
                            self._get_categories_for_feature(
                                transformer,
                                columns,
                                column,
                            )
                        )

                    schema.append({

                        "name":
                            str(column),

                        "type":
                            inferred_type,

                        "input_type":
                            (
                                "select"
                                if inferred_type
                                == "categorical"
                                else "number"
                            ),

                        "categories":
                            categories,

                        "transformer":
                            transformer_name,

                    })

        # --------------------------------------------------------
        # Remove duplicates
        # --------------------------------------------------------

        unique_schema = []

        seen = set()

        for feature in schema:

            name = feature["name"]

            if name in seen:

                continue

            seen.add(name)

            unique_schema.append(
                feature
            )

        return unique_schema

    # ============================================================
    # Infer Transformer Type
    # ============================================================

    @staticmethod
    def _infer_transformer_type(
        transformer: Any,
    ) -> str:
        """
        Infer whether a transformer is numerical
        or categorical.
        """

        steps = getattr(
            transformer,
            "steps",
            [],
        )

        for (
            step_name,
            step,
        ) in steps:

            class_name = (
                step.__class__.__name__
                .lower()
            )

            if (
                "onehot" in class_name
                or "ordinal" in class_name
            ):

                return "categorical"

        return "numerical"

    # ============================================================
    # Get Categories For Feature
    # ============================================================

    @staticmethod
    def _get_categories_for_feature(
        transformer: Any,
        columns: List[Any],
        feature_name: str,
    ) -> List[Any]:
        """
        Extract categories belonging specifically
        to one categorical feature.

        This is important when multiple categorical
        columns are handled by the same OneHotEncoder.
        """

        # --------------------------------------------------------
        # Find encoder
        # --------------------------------------------------------

        encoder = None

        direct_categories = getattr(
            transformer,
            "categories_",
            None,
        )

        if direct_categories is not None:

            encoder = transformer

        else:

            steps = getattr(
                transformer,
                "steps",
                [],
            )

            for (
                step_name,
                step,
            ) in steps:

                step_categories = getattr(
                    step,
                    "categories_",
                    None,
                )

                if step_categories is not None:

                    encoder = step

                    break

        if encoder is None:

            return []

        categories = getattr(
            encoder,
            "categories_",
            None,
        )

        if categories is None:

            return []

        # --------------------------------------------------------
        # Locate feature index
        # --------------------------------------------------------

        try:

            feature_index = columns.index(
                feature_name
            )

        except ValueError:

            return []

        if (
            feature_index
            >= len(categories)
        ):

            return []

        return [

            value

            for value
            in list(
                categories[
                    feature_index
                ]
            )

            if value is not None

        ]

    # ============================================================
    # Get Input Schema
    # ============================================================

    def get_input_schema(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Return the original input feature schema.
        """

        if self.model is None:

            self.load_artifacts()

        return self.feature_schema

    # ============================================================
    # Get Feature Names
    # ============================================================

    def get_feature_names(
        self,
    ) -> List[str]:
        """
        Return names of all original input features.
        """

        schema = (
            self.get_input_schema()
        )

        return [

            feature["name"]

            for feature in schema

        ]

    # ============================================================
    # Get Numerical Features
    # ============================================================

    def get_numerical_features(
        self,
    ) -> List[str]:
        """
        Return numerical input features.
        """

        schema = (
            self.get_input_schema()
        )

        return [

            feature["name"]

            for feature in schema

            if feature["type"]
            == "numerical"

        ]

    # ============================================================
    # Get Categorical Features
    # ============================================================

    def get_categorical_features(
        self,
    ) -> List[str]:
        """
        Return categorical input features.
        """

        schema = (
            self.get_input_schema()
        )

        return [

            feature["name"]

            for feature in schema

            if feature["type"]
            == "categorical"

        ]

    # ============================================================
    # Get Feature Categories
    # ============================================================

    def get_feature_categories(
        self,
        feature_name: str,
    ) -> List[Any]:
        """
        Return possible values for a categorical feature.
        """

        schema = (
            self.get_input_schema()
        )

        for feature in schema:

            if (
                feature["name"]
                == feature_name
            ):

                return feature.get(
                    "categories",
                    [],
                )

        raise ValueError(
            f"Unknown feature: "
            f"{feature_name}"
        )

    # ============================================================
    # Validate Input
    # ============================================================

    def validate_input(
        self,
        input_data: Dict[str, Any],
    ) -> None:
        """
        Validate prediction input against
        the discovered feature schema.
        """

        if not isinstance(
            input_data,
            dict,
        ):

            raise TypeError(
                "Prediction input must be "
                "a dictionary."
            )

        schema = (
            self.get_input_schema()
        )

        expected_features = {

            feature["name"]

            for feature in schema

        }

        provided_features = set(
            input_data.keys()
        )

        # --------------------------------------------------------
        # Missing features
        # --------------------------------------------------------

        missing_features = (
            expected_features
            - provided_features
        )

        if missing_features:

            raise ValueError(
                "Missing prediction features: "
                + ", ".join(
                    sorted(
                        missing_features
                    )
                )
            )

        # --------------------------------------------------------
        # Unexpected features
        # --------------------------------------------------------

        extra_features = (
            provided_features
            - expected_features
        )

        if extra_features:

            raise ValueError(
                "Unexpected prediction features: "
                + ", ".join(
                    sorted(
                        extra_features
                    )
                )
            )

        # --------------------------------------------------------
        # Individual feature validation
        # --------------------------------------------------------

        for feature in schema:

            name = feature["name"]

            feature_type = feature["type"]

            value = input_data[name]

            # ----------------------------------------------------
            # None
            # ----------------------------------------------------

            if value is None:

                raise ValueError(
                    f"Feature '{name}' "
                    f"cannot be None."
                )

            # ----------------------------------------------------
            # Empty string
            # ----------------------------------------------------

            if (
                isinstance(
                    value,
                    str,
                )
                and not value.strip()
            ):

                raise ValueError(
                    f"Feature '{name}' "
                    f"cannot be empty."
                )

            # ----------------------------------------------------
            # Numerical
            # ----------------------------------------------------

            if (
                feature_type
                == "numerical"
            ):

                try:

                    numeric_value = float(
                        value
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    raise ValueError(
                        f"Feature '{name}' "
                        f"must be numerical."
                    )

                if pd.isna(
                    numeric_value
                ):

                    raise ValueError(
                        f"Feature '{name}' "
                        f"cannot be NaN."
                    )

            # ----------------------------------------------------
            # Categorical
            # ----------------------------------------------------

            elif (
                feature_type
                == "categorical"
            ):

                categories = (
                    feature.get(
                        "categories",
                        [],
                    )
                )

                if (
                    categories
                    and value not in categories
                ):

                    raise ValueError(
                        f"Invalid value "
                        f"'{value}' for "
                        f"feature '{name}'. "
                        f"Expected one of: "
                        f"{categories}"
                    )

    # ============================================================
    # Prepare Input
    # ============================================================

    def _prepare_input(
        self,
        input_data: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Convert raw user input into the exact
        DataFrame structure expected by the
        fitted preprocessor.
        """

        feature_names = (
            self.get_feature_names()
        )

        row = {}

        for feature in feature_names:

            value = input_data[
                feature
            ]

            # ----------------------------------------------------
            # Convert numerical values
            # ----------------------------------------------------

            feature_info = next(

                item

                for item
                in self.feature_schema

                if item["name"]
                == feature

            )

            if (
                feature_info["type"]
                == "numerical"
            ):

                value = float(
                    value
                )

            row[feature] = value

        return pd.DataFrame(
            [row],
            columns=feature_names,
        )

    # ============================================================
    # Prediction
    # ============================================================

    def predict(
        self,
        input_data: Dict[str, Any],
    ) -> Any:
        """
        Generate a prediction from raw feature values.
        """

        if self.model is None:

            self.load_artifacts()

        # --------------------------------------------------------
        # Validate input
        # --------------------------------------------------------

        self.validate_input(
            input_data
        )

        # --------------------------------------------------------
        # Prepare DataFrame
        # --------------------------------------------------------

        input_df = (
            self._prepare_input(
                input_data
            )
        )

        # --------------------------------------------------------
        # Apply training preprocessing
        # --------------------------------------------------------

        transformed_input = (
            self.preprocessor.transform(
                input_df
            )
        )

        # --------------------------------------------------------
        # Generate prediction
        # --------------------------------------------------------

        prediction = (
            self.model.predict(
                transformed_input
            )
        )

        # --------------------------------------------------------
        # Return first prediction
        # --------------------------------------------------------

        return prediction[0]

    # ============================================================
    # Model Name
    # ============================================================

    def get_model_name(
        self,
    ) -> str:
        """
        Return the trained model name.
        """

        if self.metadata:

            return self.metadata.get(
                "model_name",
                (
                    self.model.__class__.__name__
                    if self.model is not None
                    else "Unknown"
                ),
            )

        if self.model is not None:

            return (
                self.model.__class__.__name__
            )

        return "Unknown"

    # ============================================================
    # Problem Type
    # ============================================================

    def get_problem_type(
        self,
    ) -> str:
        """
        Return the ML problem type.
        """

        if self.metadata:

            return self.metadata.get(
                "problem_type",
                "UNKNOWN",
            )

        return "UNKNOWN"

    # ============================================================
    # Target Column
    # ============================================================

    def get_target_column(
        self,
    ) -> str:
        """
        Return the target column stored
        in model metadata.
        """

        if self.metadata:

            return self.metadata.get(
                "target_column",
                "Unknown",
            )

        return "Unknown"

    # ============================================================
    # Primary Metric
    # ============================================================

    def get_primary_metric(
        self,
    ) -> str:
        """
        Return the metric used for model evaluation.
        """

        if self.metadata:

            return self.metadata.get(
                "primary_metric",
                "Unknown",
            )

        return "Unknown"

    # ============================================================
    # Model Score
    # ============================================================

    def get_model_score(
        self,
    ) -> Any:
        """
        Return the best model evaluation score.
        """

        if self.metadata:

            return self.metadata.get(
                "best_model_score"
            )

        return None

    # ============================================================
    # Prediction Information
    # ============================================================

    def get_prediction_info(
        self,
    ) -> Dict[str, Any]:
        """
        Return all information required by
        the generic ArchForge prediction UI.
        """

        if self.model is None:

            self.load_artifacts()

        schema = (
            self.get_input_schema()
        )

        return {

            "model_name":
                self.get_model_name(),

            "problem_type":
                self.get_problem_type(),

            "target_column":
                self.get_target_column(),

            "primary_metric":
                self.get_primary_metric(),

            "model_score":
                self.get_model_score(),

            "features":
                schema,

            "feature_names":
                self.get_feature_names(),

            "numerical_features":
                self.get_numerical_features(),

            "categorical_features":
                self.get_categorical_features(),

            "feature_count":
                len(schema),

        }