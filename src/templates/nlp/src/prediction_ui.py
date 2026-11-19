
"""
prediction_ui.py

ArchForge Generic NLP Streamlit Prediction Interface.

Features:
1. Dynamically displays classification labels.
2. Supports binary and multiclass classification.
3. Displays selected best model and validation metrics.
4. Displays prediction confidence when supported.
5. Displays class probabilities when supported.
6. Reloads the model when model package files change.
7. Reads model information from metadata.json.
"""

from pathlib import Path
import sys

import streamlit as st


# Resolve generated project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.prediction_pipeline import NLPPredictionPipeline


st.set_page_config(
    page_title="ArchForge NLP Prediction",
    page_icon="🧠",
    layout="centered",
)


# Model package paths.
MODEL_DIR = PROJECT_ROOT / "artifacts" / "final_model"

MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


# Generate a cache version from packaged model files.
def get_model_version():

    paths = [
        MODEL_PATH,
        VECTORIZER_PATH,
        METADATA_PATH,
    ]

    if not all(path.is_file() for path in paths):
        return None

    return tuple(
        (
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in paths
    )


# Load prediction pipeline.
@st.cache_resource
def load_prediction_pipeline(project_root, model_version):

    return NLPPredictionPipeline(
        Path(project_root)
    )


# Application header.
st.title("🧠 ArchForge NLP Prediction")

st.markdown(
    """
    Analyze text using your trained NLP classification model.

    Enter text below to generate a prediction.
    """
)

st.divider()


# Check model availability.
if not all(
    path.is_file()
    for path in [
        MODEL_PATH,
        VECTORIZER_PATH,
        METADATA_PATH,
    ]
):

    st.error(
        "Trained model package not found. "
        "Please add a dataset and run the training pipeline."
    )

    st.stop()


# Load latest model.
try:

    model_version = get_model_version()

    prediction_pipeline = load_prediction_pipeline(
        str(PROJECT_ROOT),
        model_version,
    )

    metadata = prediction_pipeline.metadata

except Exception as error:

    st.error(f"Unable to load model: {error}")
    st.stop()


# Read dynamic model information.
target_column = metadata.get(
    "target_column",
    "target",
)

text_columns = metadata.get(
    "text_columns",
    [],
)

best_model = metadata.get(
    "best_model",
    "Not available",
)

validation_accuracy = metadata.get(
    "validation_accuracy",
)

validation_f1_score = metadata.get(
    "validation_f1_score",
)

validation_method = metadata.get(
    "validation_method",
    "Not available",
)


# Detect classification labels from the trained model.
trained_model = getattr(
    prediction_pipeline,
    "model",
    None,
)

model_classes = getattr(
    trained_model,
    "classes_",
    None,
)

if model_classes is not None:
    model_classes = [str(label) for label in model_classes]
else:
    model_classes = []


# Model information.
st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Best Model",
        best_model,
    )

with col2:

    st.metric(
        "Validation Accuracy",
        (
            f"{validation_accuracy * 100:.2f}%"
            if validation_accuracy is not None
            else "Not available"
        ),
    )


if validation_f1_score is not None:

    st.metric(
        "Validation Weighted F1-score",
        f"{validation_f1_score * 100:.2f}%",
    )


with st.expander("Dataset and training details"):

    st.write(f"**Prediction target:** {target_column}")

    st.write(
        "**Text columns:** "
        + (
            ", ".join(text_columns)
            if text_columns
            else "Not available"
        )
    )

    st.write(
        f"**Validation method:** {validation_method}"
    )

    st.write("**Classification labels:**")

    if model_classes:

        st.write(" | ".join(model_classes))

    else:

        st.caption(
            "Labels will be displayed after prediction "
            "if provided by the prediction pipeline."
        )


st.divider()


# Input section.
st.subheader("Enter your text")

input_text = st.text_area(
    "Text input",
    placeholder="Enter text here for prediction...",
    height=180,
    label_visibility="collapsed",
)


# Prediction.
if st.button(
    "🔮 Predict",
    type="primary",
    use_container_width=True,
):

    if not input_text.strip():

        st.warning(
            "Please enter some text before predicting."
        )

    else:

        try:

            with st.spinner("Analyzing text..."):

                result = prediction_pipeline.predict(
                    input_text
                )

            prediction = result["prediction"]

            confidence = result.get("confidence")

            probabilities = result.get(
                "class_probabilities",
                {},
            )

            st.divider()

            st.subheader("Prediction Result")

            # Display predicted class without assuming its meaning.
            st.success(
                f"Predicted {target_column}: {prediction}"
            )

            # Prediction confidence.
            if confidence is not None:

                st.metric(
                    "Prediction Confidence",
                    f"{confidence * 100:.2f}%",
                )

            # Dynamically display available class probabilities.
            if probabilities:

                st.subheader("Class Probabilities")

                for label, probability in probabilities.items():

                    probability = float(probability)

                    percentage = probability * 100

                    st.write(
                        f"**{label}** — {percentage:.2f}%"
                    )

                    st.progress(
                        min(
                            max(probability, 0.0),
                            1.0,
                        )
                    )

            else:

                st.caption(
                    "Class probabilities are not available "
                    "for the selected model."
                )

            # Submitted text.
            with st.expander("View submitted text"):

                st.write(input_text)

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


st.divider()

st.caption(
    "Powered by ArchForge | Generic NLP Classification Pipeline"
)
