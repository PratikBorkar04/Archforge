"""
prediction_pipeline.py

Prediction component for ArchForge.

Responsibilities
----------------
1. Load the trained model.
2. Load the fitted preprocessor.
3. Discover the original input feature schema.
4. Identify numerical and categorical features.
5. Provide categorical values for UI generation.
6. Validate user input.
7. Apply the same preprocessing used during training.
8. Generate predictions.

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

        self.model_path = (
            self.project_root
            / "artifacts"
            / "models"
            / "best_model.pkl"
        )

        self.preprocessor_path = (
            self.project_root
            / "artifacts"
            / "preprocessor.pkl"
        )

        self.metadata_path = (
            self.project_root
            / "artifacts"
            / "models"
            / "model_metadata.json"
        )

        self.model = None
        self.preprocessor = None
        self.metadata = None

        self.feature_schema = None

    # ============================================================
    # Load Artifacts
    # ============================================================

    def load_artifacts(
        self,
    ) -> None:
        """
        Load model, preprocessor and metadata.
        """

        # --------------------------------------------------------
        # Validate artifact paths
        # --------------------------------------------------------

        if not self.model_path.exists():

            raise FileNotFoundError(
                "Trained model artifact was not found:\n"
                f"{self.model_path}"
            )

        if not self.preprocessor_path.exists():

            raise FileNotFoundError(
                "Preprocessor artifact was not found:\n"
                f"{self.preprocessor_path}"
            )

        # --------------------------------------------------------
        # Load model
        # --------------------------------------------------------

        with open(
            self.model_path,
            "rb",
        ) as file:

            self.model = pickle.load(
                file
            )

        # --------------------------------------------------------
        # Load preprocessor
        # --------------------------------------------------------

        with open(
            self.preprocessor_path,
            "rb",
        ) as file:

            self.preprocessor = pickle.load(
                file
            )

        # --------------------------------------------------------
        # Load metadata
        # --------------------------------------------------------

        if self.metadata_path.exists():

            with open(
                self.metadata_path,
                "r",
                encoding="utf-8",
            ) as file:

                self.metadata = json.load(
                    file
                )

        else:

            self.metadata = {}

        # --------------------------------------------------------
        # Build feature schema
        # --------------------------------------------------------

        self.feature_schema = (
            self._build_feature_schema()
        )

        print(
            "\nPrediction artifacts loaded successfully."
        )

        print(
            f"Model: "
            f"{self.get_model_name()}"
        )

        print(
            f"Problem type: "
            f"{self.get_problem_type()}"
        )

        print(
            f"Input features: "
            f"{len(self.feature_schema)}"
        )

    # ============================================================
    # Build Feature Schema
    # ============================================================

    def _build_feature_schema(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Discover the original input feature schema
        from the fitted preprocessor.

        Returns
        -------
        List[Dict[str, Any]]
            Feature definitions for the prediction UI.
        """

        if self.preprocessor is None:

            raise RuntimeError(
                "Preprocessor must be loaded "
                "before building feature schema."
            )

        schema = []

        # --------------------------------------------------------
        # Locate ColumnTransformer transformers
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Inspect each transformer
        # --------------------------------------------------------

        for (
            transformer_name,
            transformer,
            columns,
        ) in transformers:

            # Ignore dropped columns
            if transformer == "drop":

                continue

            # Ignore empty column groups
            if columns is None:

                continue

            # Convert column selectors to list
            if isinstance(
                columns,
                str,
            ):

                columns = [columns]

            else:

                try:

                    columns = list(
                        columns
                    )

                except TypeError:

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

                for column in columns:

                    schema.append({

                        "name":
                            str(column),

                        "type":
                            "numerical",

                        "input_type":
                            "number",

                        "transformer":
                            transformer_name,

                    })

            # ----------------------------------------------------
            # Categorical transformer
            # ----------------------------------------------------

            elif transformer_name in {
                "cat",
                "categorical",
                "categorical_pipeline",
            }:

                categories = (
                    self._get_categories(
                        transformer
                    )
                )

                for column in columns:

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
            # Unknown transformer
            # ----------------------------------------------------

            else:

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
                        self._get_categories(
                            transformer
                        )
                    )

                for column in columns:

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
        # Remove duplicate features
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
        Infer whether an unknown transformer is
        numerical or categorical.
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
    # Get Categories
    # ============================================================

    @staticmethod
    def _get_categories(
        transformer: Any,
    ) -> List[Any]:
        """
        Extract categorical values from a fitted
        OneHotEncoder or similar transformer.
        """

        categories = []

        # --------------------------------------------------------
        # Direct categories_
        # --------------------------------------------------------

        direct_categories = getattr(
            transformer,
            "categories_",
            None,
        )

        if direct_categories is not None:

            for category_group in (
                direct_categories
            ):

                categories.extend(
                    list(
                        category_group
                    )
                )

            return [
                value
                for value in categories
                if value is not None
            ]

        # --------------------------------------------------------
        # Pipeline containing encoder
        # --------------------------------------------------------

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

                for category_group in (
                    step_categories
                ):

                    categories.extend(
                        list(
                            category_group
                        )
                    )

                break

        return [
            value
            for value in categories
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

        This method is intended to be consumed by
        the generic ArchForge prediction UI.
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
        Return names of all input features.
        """

        schema = self.get_input_schema()

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

        schema = self.get_input_schema()

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

        schema = self.get_input_schema()

        return [

            feature["name"]

            for feature in schema

            if feature["type"]
            == "categorical"

        ]

    # ============================================================
    # Get Categorical Values
    # ============================================================

    def get_feature_categories(
        self,
        feature_name: str,
    ) -> List[Any]:
        """
        Return possible values for a categorical feature.
        """

        schema = self.get_input_schema()

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

        schema = self.get_input_schema()

        expected_features = {
            feature["name"]
            for feature in schema
        }

        provided_features = set(
            input_data.keys()
        )

        missing_features = (
            expected_features
            - provided_features
        )

        extra_features = (
            provided_features
            - expected_features
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
        # Validate individual values
        # --------------------------------------------------------

        for feature in schema:

            name = feature["name"]
            feature_type = feature["type"]

            value = input_data[name]

            if value is None:

                raise ValueError(
                    f"Feature '{name}' "
                    f"cannot be None."
                )

            if (
                feature_type
                == "numerical"
            ):

                try:

                    float(value)

                except (
                    TypeError,
                    ValueError,
                ):

                    raise ValueError(
                        f"Feature '{name}' "
                        f"must be numerical."
                    )

            elif (
                feature_type
                == "categorical"
            ):

                categories = feature.get(
                    "categories",
                    [],
                )

                if (
                    categories
                    and value not in categories
                ):

                    raise ValueError(
                        f"Invalid value '{value}' "
                        f"for feature '{name}'. "
                        f"Expected one of: "
                        f"{categories}"
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

        Parameters
        ----------
        input_data:
            Dictionary containing original feature values.

        Returns
        -------
        Any
            Prediction generated by the trained model.
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
        # Create DataFrame
        # --------------------------------------------------------

        feature_names = (
            self.get_feature_names()
        )

        input_df = pd.DataFrame(
            [
                {
                    feature:
                        input_data[feature]
                    for feature in feature_names
                }
            ]
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

        return prediction[0]

    # ============================================================
    # Model Name
    # ============================================================

    def get_model_name(
        self,
    ) -> str:

        if self.metadata:

            return self.metadata.get(
                "model_name",
                self.model.__class__.__name__
                if self.model is not None
                else "Unknown",
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
        Return target column when available.
        """

        if self.metadata:

            return self.metadata.get(
                "target_column",
                "Unknown",
            )

        return "Unknown"

    # ============================================================
    # Prediction Information
    # ============================================================

    def get_prediction_info(
        self,
    ) -> Dict[str, Any]:
        """
        Return information required by the UI.
        """

        if self.model is None:

            self.load_artifacts()

        return {

            "model_name":
                self.get_model_name(),

            "problem_type":
                self.get_problem_type(),

            "target_column":
                self.get_target_column(),

            "features":
                self.get_input_schema(),

            "feature_count":
                len(
                    self.get_input_schema()
                ),

        }