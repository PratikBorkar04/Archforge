"""
prediction_ui.py

Professional generic prediction UI for ArchForge.

Responsibilities
----------------
1. Load trained model.
2. Load fitted preprocessor.
3. Load model metadata.
4. Discover original input features.
5. Correctly detect numerical/categorical features.
6. Extract categories independently for each categorical feature.
7. Create professional Streamlit input controls.
8. Apply the fitted preprocessor.
9. Generate predictions.
10. Display model performance information.

This module contains NO dataset-specific feature definitions.
"""

from pathlib import Path
from typing import Dict, Any, List

import json
import joblib
import pandas as pd
import streamlit as st


class PredictionUI:
    """
    Generic professional Streamlit prediction interface
    for ArchForge.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        project_root: Path,
    ) -> None:

        self.project_root = Path(
            project_root
        ).resolve()

        self.model_dir = (
            self.project_root
            / "artifacts"
            / "models"
        )

        self.model_path = (
            self.model_dir
            / "best_model.pkl"
        )

        self.metadata_path = (
            self.model_dir
            / "model_metadata.json"
        )

        self.preprocessor_path = (
            self.project_root
            / "artifacts"
            / "preprocessor.pkl"
        )

        self.model = None
        self.preprocessor = None
        self.metadata = {}

        self.feature_schema = []

    # ============================================================
    # LOAD ARTIFACTS
    # ============================================================

    def load_artifacts(self) -> None:
        """
        Load trained model, preprocessor and metadata.
        """

        if not self.model_path.exists():

            raise FileNotFoundError(
                "Model artifact not found:\n"
                f"{self.model_path}"
            )

        if not self.preprocessor_path.exists():

            raise FileNotFoundError(
                "Preprocessor artifact not found:\n"
                f"{self.preprocessor_path}"
            )

        if not self.metadata_path.exists():

            raise FileNotFoundError(
                "Model metadata not found:\n"
                f"{self.metadata_path}"
            )

        # --------------------------------------------------------
        # Load model
        # --------------------------------------------------------

        self.model = joblib.load(
            self.model_path
        )

        # --------------------------------------------------------
        # Load preprocessor
        # --------------------------------------------------------

        self.preprocessor = joblib.load(
            self.preprocessor_path
        )

        # --------------------------------------------------------
        # Load metadata
        # --------------------------------------------------------

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.metadata = json.load(
                file
            )

        # --------------------------------------------------------
        # Build feature schema
        # --------------------------------------------------------

        self.feature_schema = (
            self._build_feature_schema()
        )

    # ============================================================
    # PAGE STYLING
    # ============================================================

    @staticmethod
    def _apply_page_style() -> None:
        """
        Apply professional styling to the Streamlit UI.
        """

        st.markdown(
            """
            <style>

            /* ------------------------------------------------
               Main page
            ------------------------------------------------ */

            .block-container {
                max-width: 1050px;
                padding-top: 2.5rem;
                padding-bottom: 3rem;
            }

            /* ------------------------------------------------
               Header
            ------------------------------------------------ */

            .archforge-header {
                padding: 0.5rem 0 1.5rem 0;
            }

            .archforge-brand {
                font-size: 2.2rem;
                font-weight: 700;
                letter-spacing: -0.5px;
                margin-bottom: 0.2rem;
            }

            .archforge-description {
                font-size: 1rem;
                opacity: 0.65;
                margin-bottom: 0.5rem;
            }

            /* ------------------------------------------------
               Cards
            ------------------------------------------------ */

            .info-card {
                padding: 1.1rem 1.2rem;
                border: 1px solid rgba(128, 128, 128, 0.20);
                border-radius: 12px;
                background: rgba(128, 128, 128, 0.035);
                margin-bottom: 1rem;
            }

            .info-label {
                font-size: 0.78rem;
                opacity: 0.60;
                margin-bottom: 0.25rem;
            }

            .info-value {
                font-size: 1.05rem;
                font-weight: 600;
            }

            /* ------------------------------------------------
               Prediction result
            ------------------------------------------------ */

            .prediction-card {
                padding: 1.5rem;
                border: 1px solid rgba(128, 128, 128, 0.20);
                border-radius: 14px;
                background: rgba(128, 128, 128, 0.035);
                margin-top: 1.2rem;
                margin-bottom: 1rem;
            }

            .prediction-label {
                font-size: 0.85rem;
                opacity: 0.60;
                margin-bottom: 0.35rem;
            }

            .prediction-value {
                font-size: 2.2rem;
                font-weight: 700;
            }

            /* ------------------------------------------------
               Input section
            ------------------------------------------------ */

            .section-title {
                font-size: 1.25rem;
                font-weight: 650;
                margin-top: 0.5rem;
                margin-bottom: 0.3rem;
            }

            .section-description {
                font-size: 0.9rem;
                opacity: 0.60;
                margin-bottom: 1rem;
            }

            /* ------------------------------------------------
               Footer
            ------------------------------------------------ */

            .archforge-footer {
                text-align: center;
                opacity: 0.45;
                font-size: 0.8rem;
                margin-top: 2rem;
            }

            </style>
            """,
            unsafe_allow_html=True,
        )

    # ============================================================
    # MAIN UI
    # ============================================================

    def run(self) -> None:
        """
        Start the Streamlit prediction UI.
        """

        # --------------------------------------------------------
        # Load artifacts
        # --------------------------------------------------------

        try:

            self.load_artifacts()

        except Exception as error:

            st.error(
                "Unable to load prediction artifacts."
            )

            st.exception(
                error
            )

            return

        # --------------------------------------------------------
        # Page configuration
        # --------------------------------------------------------

        st.set_page_config(
            page_title="ArchForge Prediction",
            page_icon="🤖",
            layout="centered",
        )

        self._apply_page_style()

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        model_name = self.metadata.get(
            "model_name",
            self.model.__class__.__name__,
        )

        problem_type = self.metadata.get(
            "problem_type",
            "UNKNOWN",
        )

        target_column = self.metadata.get(
            "target_column",
            "Prediction",
        )

        feature_count = len(
            self.feature_schema
        )

        best_model_score = self.metadata.get(
            "best_model_score"
        )

        primary_metric = self.metadata.get(
            "primary_metric"
        )

        # ========================================================
        # HEADER
        # ========================================================

        st.markdown(
            """
            <div class="archforge-header">
                <div class="archforge-brand">
                    🤖 ArchForge
                </div>
                <div class="archforge-description">
                    AI-Powered Prediction Interface
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ========================================================
        # MODEL OVERVIEW
        # ========================================================

        st.markdown(
            '<div class="section-title">Model Overview</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Information about the trained model currently used for prediction.'
            '</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">MODEL</div>
                    <div class="info-value">{model_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">PROBLEM TYPE</div>
                    <div class="info-value">{problem_type}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">PREDICTION TARGET</div>
                    <div class="info-value">{target_column}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">INPUT FEATURES</div>
                    <div class="info-value">{feature_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ========================================================
        # MODEL PERFORMANCE
        # ========================================================

        if (
            best_model_score is not None
            and primary_metric
        ):

            st.markdown(
                '<div class="section-title">Model Performance</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-description">'
                'Performance of the selected model on the evaluation dataset.'
                '</div>',
                unsafe_allow_html=True,
            )

            score_col1, score_col2 = st.columns(2)

            with score_col1:

                metric_display = (
                    str(primary_metric)
                    .replace("_", " ")
                    .upper()
                )

                st.metric(
                    metric_display,
                    f"{float(best_model_score):.4f}",
                )

            with score_col2:

                if (
                    problem_type.upper()
                    == "REGRESSION"
                    and primary_metric
                    == "r2_score"
                ):

                    percentage = (
                        float(best_model_score)
                        * 100
                    )

                    st.metric(
                        "R² SCORE",
                        f"{percentage:.2f}%",
                    )

                else:

                    st.metric(
                        "MODEL SCORE",
                        f"{float(best_model_score):.4f}",
                    )

        st.divider()

        # ========================================================
        # INPUT SECTION
        # ========================================================

        st.markdown(
            '<div class="section-title">Prediction Inputs</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Enter the feature values required by the trained model.'
            '</div>',
            unsafe_allow_html=True,
        )

        if not self.feature_schema:

            st.error(
                "No input features were discovered "
                "from the fitted preprocessor."
            )

            return

        input_values = {}

        # --------------------------------------------------------
        # Create inputs in two-column layout
        # --------------------------------------------------------

        columns = st.columns(2)

        for index, feature in enumerate(
            self.feature_schema
        ):

            feature_name = feature["name"]

            with columns[index % 2]:

                input_values[
                    feature_name
                ] = self._create_input(
                    feature
                )

        st.divider()

        # ========================================================
        # PREDICTION BUTTON
        # ========================================================

        predict_clicked = st.button(
            "🔮 Generate Prediction",
            type="primary",
            use_container_width=True,
        )

        if predict_clicked:

            self._make_prediction(
                input_values,
                target_column,
            )

        # ========================================================
        # FOOTER
        # ========================================================

        st.markdown(
            """
            <div class="archforge-footer">
                Powered by ArchForge ML Pipeline
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ============================================================
    # BUILD FEATURE SCHEMA
    # ============================================================

    def _build_feature_schema(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Discover original input features from
        the fitted ColumnTransformer.

        IMPORTANT:
        Categories are extracted independently for
        each categorical feature.
        """

        if self.preprocessor is None:

            raise RuntimeError(
                "Preprocessor has not been loaded."
            )

        transformers = getattr(
            self.preprocessor,
            "transformers_",
            None,
        )

        if transformers is None:

            raise RuntimeError(
                "Fitted transformer information "
                "was not found."
            )

        schema = []

        # --------------------------------------------------------
        # Inspect transformers
        # --------------------------------------------------------

        for (
            transformer_name,
            transformer,
            columns,
        ) in transformers:

            if transformer == "drop":

                continue

            if transformer_name == "remainder":

                continue

            if columns is None:

                continue

            # ----------------------------------------------------
            # Convert columns to list
            # ----------------------------------------------------

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
            # Detect feature type
            # ----------------------------------------------------

            feature_type = (
                self._infer_transformer_type(
                    transformer_name,
                    transformer,
                )
            )

            # ----------------------------------------------------
            # Add each feature separately
            # ----------------------------------------------------

            for column_index, column in enumerate(
                columns
            ):

                categories = []

                if feature_type == "categorical":

                    categories = (
                        self._get_categories_for_column(
                            transformer=transformer,
                            column_index=column_index,
                        )
                    )

                schema.append(
                    {
                        "name": str(column),

                        "type": feature_type,

                        "categories": categories,

                        "transformer": str(
                            transformer_name
                        ),
                    }
                )

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
    # INFER TRANSFORMER TYPE
    # ============================================================

    @staticmethod
    def _infer_transformer_type(
        transformer_name: str,
        transformer: Any,
    ) -> str:
        """
        Determine whether the transformer handles
        numerical or categorical features.
        """

        name = str(
            transformer_name
        ).lower()

        # --------------------------------------------------------
        # Transformer name
        # --------------------------------------------------------

        if any(
            word in name
            for word in [
                "categor",
                "ordinal",
                "onehot",
                "cat",
            ]
        ):

            return "categorical"

        if any(
            word in name
            for word in [
                "numeric",
                "numerical",
                "num",
            ]
        ):

            return "numerical"

        # --------------------------------------------------------
        # Pipeline steps
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

            class_name = (
                step.__class__.__name__
                .lower()
            )

            if (
                "onehot" in class_name
                or "ordinal" in class_name
            ):

                return "categorical"

        # --------------------------------------------------------
        # Direct transformer
        # --------------------------------------------------------

        transformer_class = (
            transformer.__class__.__name__
            .lower()
        )

        if (
            "onehot" in transformer_class
            or "ordinal" in transformer_class
        ):

            return "categorical"

        return "numerical"

    # ============================================================
    # GET CATEGORIES FOR ONE COLUMN
    # ============================================================

    @staticmethod
    def _get_categories_for_column(
        transformer: Any,
        column_index: int,
    ) -> List[Any]:
        """
        Extract categories ONLY for the requested
        categorical column.

        This fixes the previous problem where categories
        from Fuel_Type, Seller_Type, Transmission and
        Owner were combined into one dropdown.
        """

        # --------------------------------------------------------
        # Direct transformer
        # --------------------------------------------------------

        direct_categories = getattr(
            transformer,
            "categories_",
            None,
        )

        if direct_categories is not None:

            if (
                column_index
                < len(direct_categories)
            ):

                return [
                    value
                    for value in list(
                        direct_categories[
                            column_index
                        ]
                    )
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

                if (
                    column_index
                    < len(step_categories)
                ):

                    return [
                        value
                        for value in list(
                            step_categories[
                                column_index
                            ]
                        )
                        if value is not None
                    ]

        return []

    # ============================================================
    # CREATE INPUT
    # ============================================================

    def _create_input(
        self,
        feature: Dict[str, Any],
    ) -> Any:
        """
        Create a professional Streamlit input.
        """

        feature_name = feature[
            "name"
        ]

        feature_type = feature[
            "type"
        ]

        # --------------------------------------------------------
        # Numerical input
        # --------------------------------------------------------

        if feature_type == "numerical":

            # Detect common integer-style numerical fields
            # without affecting decimal-valued features.
            integer_style = (
                self._is_integer_style_feature(
                    feature_name
                )
            )

            if integer_style:

                return st.number_input(
                    feature_name,
                    value=0,
                    step=1,
                    format="%d",
                    key=(
                        "prediction_"
                        + feature_name
                    ),
                )

            return st.number_input(
                feature_name,
                value=0.0,
                step=0.01,
                format="%.2f",
                key=(
                    "prediction_"
                    + feature_name
                ),
            )

        # --------------------------------------------------------
        # Categorical input
        # --------------------------------------------------------

        categories = feature.get(
            "categories",
            [],
        )

        if categories:

            return st.selectbox(
                feature_name,
                options=categories,
                key=(
                    "prediction_"
                    + feature_name
                ),
            )

        # --------------------------------------------------------
        # Unknown categorical
        # --------------------------------------------------------

        return st.text_input(
            feature_name,
            key=(
                "prediction_"
                + feature_name
            ),
        )

    # ============================================================
    # INTEGER-STYLE FEATURE DETECTION
    # ============================================================

    @staticmethod
    def _is_integer_style_feature(
        feature_name: str,
    ) -> bool:
        """
        Identify common count/year style fields.

        These are displayed without unnecessary decimals.
        """

        name = str(
            feature_name
        ).strip().lower()

        integer_keywords = (
            "year",
            "years",
            "count",
            "quantity",
            "qty",
            "number",
            "num",
            "age",
            "owner",
            "kms",
            "km",
            "mileage",
            "days",
            "months",
            "units",
        )

        return any(
            keyword in name
            for keyword in integer_keywords
        )

    # ============================================================
    # VALIDATE INPUT
    # ============================================================

    def _validate_input(
        self,
        input_values: Dict[str, Any],
    ) -> None:
        """
        Validate user input before prediction.
        """

        expected_features = {
            feature["name"]
            for feature in self.feature_schema
        }

        provided_features = set(
            input_values.keys()
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
        # Extra features
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
        # Individual validation
        # --------------------------------------------------------

        for feature in self.feature_schema:

            name = feature[
                "name"
            ]

            feature_type = feature[
                "type"
            ]

            value = input_values[
                name
            ]

            # ----------------------------------------------------
            # Empty categorical text
            # ----------------------------------------------------

            if (
                feature_type
                == "categorical"
                and isinstance(
                    value,
                    str,
                )
                and not value.strip()
            ):

                raise ValueError(
                    f"Feature '{name}' "
                    "cannot be empty."
                )

            # ----------------------------------------------------
            # Numerical
            # ----------------------------------------------------

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
                        "must be numerical."
                    )

            # ----------------------------------------------------
            # Categorical
            # ----------------------------------------------------

            if (
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
                        f"Invalid value for "
                        f"feature '{name}'."
                    )

    # ============================================================
    # MAKE PREDICTION
    # ============================================================

    def _make_prediction(
        self,
        input_values: Dict[str, Any],
        target_column: str,
    ) -> None:
        """
        Generate prediction from raw user input.
        """

        try:

            # ----------------------------------------------------
            # Validate
            # ----------------------------------------------------

            self._validate_input(
                input_values
            )

            # ----------------------------------------------------
            # Preserve feature order
            # ----------------------------------------------------

            feature_names = [
                feature["name"]
                for feature
                in self.feature_schema
            ]

            # ----------------------------------------------------
            # Build DataFrame
            # ----------------------------------------------------

            input_data = pd.DataFrame(
                [
                    {
                        feature:
                            input_values[
                                feature
                            ]
                        for feature
                        in feature_names
                    }
                ]
            )

            # ----------------------------------------------------
            # Apply fitted preprocessing
            # ----------------------------------------------------

            transformed_input = (
                self.preprocessor.transform(
                    input_data
                )
            )

            # ----------------------------------------------------
            # Generate prediction
            # ----------------------------------------------------

            prediction = (
                self.model.predict(
                    transformed_input
                )
            )

            prediction_value = (
                prediction[0]
            )

            # ----------------------------------------------------
            # Display result
            # ----------------------------------------------------

            st.success(
                "Prediction generated successfully."
            )

            st.markdown(
                """
                <div class="prediction-card">
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="prediction-label">
                    PREDICTED {str(target_column).upper()}
                </div>

                <div class="prediction-value">
                    {self._format_prediction(
                        prediction_value
                    )}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

            # ----------------------------------------------------
            # Model information
            # ----------------------------------------------------

            used_model = self.metadata.get(
                "model_name",
                self.model.__class__.__name__,
            )

            evaluation_source = (
                self.metadata.get(
                    "evaluation_source"
                )
            )

            result_col1, result_col2 = (
                st.columns(2)
            )

            with result_col1:

                st.caption(
                    "Model used"
                )

                st.write(
                    str(used_model)
                )

            with result_col2:

                if evaluation_source:

                    st.caption(
                        "Evaluation dataset"
                    )

                    st.write(
                        str(
                            evaluation_source
                        ).capitalize()
                    )

        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.exception(
                error
            )

    # ============================================================
    # FORMAT PREDICTION
    # ============================================================

    @staticmethod
    def _format_prediction(
        value: Any,
    ) -> str:
        """
        Format prediction for display.
        """

        try:

            numeric_value = float(
                value
            )

            return f"{numeric_value:,.2f}"

        except (
            TypeError,
            ValueError,
        ):

            return str(value)


# ================================================================
# PUBLIC ENTRY POINT
# ================================================================

def run_prediction_ui(
    project_root: Path,
) -> None:
    """
    Public entry point for the prediction UI.
    """

    ui = PredictionUI(
        project_root=project_root
    )

    ui.run()


# ================================================================
# DIRECT STREAMLIT EXECUTION
# ================================================================

if __name__ == "__main__":

    # prediction_ui.py is inside:
    #
    # <project_root>/src/prediction_ui.py
    #
    # parent     = src
    # parent.parent = project root

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    run_prediction_ui(
        project_root=project_root
    )