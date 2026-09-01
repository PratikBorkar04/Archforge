"""
Application Entry Point
-----------------------

Main entry point of the generated ArchForge ML project.

Usage
-----
Run:

    python app.py

The application will:

    1. Execute the complete ML training pipeline.
    2. Save the trained model and preprocessing artifacts.
    3. Launch the generic Streamlit prediction UI.
"""

from pathlib import Path
import sys
import subprocess


# ------------------------------------------------------------------
# Project Root
# ------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


# ------------------------------------------------------------------
# Python Import Path
# ------------------------------------------------------------------

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ------------------------------------------------------------------
# Import Training Pipeline
# ------------------------------------------------------------------

from pipeline.training_pipeline import (
    TrainingPipeline
)


# ------------------------------------------------------------------
# Main Application
# ------------------------------------------------------------------

def main() -> None:
    """
    Execute training pipeline and launch prediction UI.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "ARCHFORGE APPLICATION"
    )

    print(
        "=" * 60
    )

    # ==============================================================
    # 1. TRAINING PIPELINE
    # ==============================================================

    print(
        "\nStarting training pipeline..."
    )

    training_pipeline = TrainingPipeline(
        project_root=PROJECT_ROOT
    )

    try:

        result = training_pipeline.run()

    except Exception as error:

        print(
            "\n" + "!" * 60
        )

        print(
            "TRAINING PIPELINE FAILED"
        )

        print(
            "!" * 60
        )

        print(
            f"\nError: {error}"
        )

        raise

    # ==============================================================
    # 2. VERIFY TRAINING RESULT
    # ==============================================================

    if not result:

        raise RuntimeError(
            "Training pipeline returned no result."
        )

    model_pushing = result.get(
        "model_pushing"
    )

    if not model_pushing:

        raise RuntimeError(
            "Model pushing information was not returned."
        )

    if model_pushing.get(
        "status"
    ) != "SUCCESS":

        raise RuntimeError(
            "Model artifacts were not created successfully."
        )

    # ==============================================================
    # 3. VERIFY REQUIRED ARTIFACTS
    # ==============================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "VERIFYING PREDICTION ARTIFACTS"
    )

    print(
        "=" * 60
    )

    model_path = (
        PROJECT_ROOT
        / "artifacts"
        / "models"
        / "best_model.pkl"
    )

    metadata_path = (
        PROJECT_ROOT
        / "artifacts"
        / "models"
        / "model_metadata.json"
    )

    preprocessor_path = (
        PROJECT_ROOT
        / "artifacts"
        / "preprocessor.pkl"
    )

    required_artifacts = {

        "Model":
            model_path,

        "Model metadata":
            metadata_path,

        "Preprocessor":
            preprocessor_path,
    }

    for (
        name,
        path,
    ) in required_artifacts.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{name} artifact not found:\n"
                f"{path}"
            )

        print(
            f"✓ {name}: {path}"
        )

    # ==============================================================
    # 4. PREDICTION UI
    # ==============================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "STARTING PREDICTION UI"
    )

    print(
        "=" * 60
    )

    # IMPORTANT:
    # prediction_ui.py is inside src/

    prediction_ui_path = (
        PROJECT_ROOT
        / "src"
        / "prediction_ui.py"
    )

    if not prediction_ui_path.exists():

        raise FileNotFoundError(
            "Prediction UI file not found:\n"
            f"{prediction_ui_path}"
        )

    print(
        "\nLaunching Streamlit..."
    )

    print(
        "Prediction UI:"
    )

    print(
        "http://localhost:8501"
    )

    print(
        "\nPress Ctrl+C to stop the application."
    )

    print(
        "=" * 60
    )

    # ==============================================================
    # 5. LAUNCH STREAMLIT
    # ==============================================================

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(prediction_ui_path),
            "--server.headless",
            "true",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":

    main()