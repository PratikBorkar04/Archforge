
"""
app.py

ArchForge NLP Application Launcher

Responsibilities:
1. Detect new or modified CSV datasets.
2. Skip training when the dataset is unchanged.
3. Automatically execute the NLP training pipeline when required.
4. Verify the newly generated model package.
5. Launch the Streamlit prediction application.
"""

from pathlib import Path
import hashlib
import json
import subprocess
import sys


def calculate_dataset_hash(data_dir: Path) -> str:
    """Generate a hash using CSV filenames and file contents."""

    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        return ""

    hasher = hashlib.sha256()

    for csv_file in csv_files:
        hasher.update(csv_file.name.encode("utf-8"))

        with csv_file.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(chunk)

    return hasher.hexdigest()


def load_previous_dataset_hash(hash_file: Path):
    """Read the hash of the previously trained dataset."""

    if not hash_file.exists():
        return None

    try:
        with hash_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data.get("dataset_hash")

    except (json.JSONDecodeError, OSError):
        return None


def save_dataset_hash(hash_file: Path, dataset_hash: str):
    """Save the hash after successful model training."""

    hash_file.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = hash_file.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(
            {"dataset_hash": dataset_hash},
            file,
            indent=4,
        )

    temporary_file.replace(hash_file)


def verify_model_package(model_dir: Path) -> bool:
    """Check whether all required model files exist."""

    required_files = [
        model_dir / "model.pkl",
        model_dir / "tfidf_vectorizer.pkl",
        model_dir / "metadata.json",
    ]

    return all(path.is_file() for path in required_files)


def train_model(project_root: Path) -> bool:
    """Execute the complete NLP training pipeline."""

    print("\n" + "=" * 60)
    print("          ARCHFORGE - NLP MODEL TRAINING")
    print("=" * 60)

    try:
        from pipeline.training_pipeline import NLPTrainingPipeline

        training_pipeline = NLPTrainingPipeline(project_root)

        result = training_pipeline.initiate_training_pipeline()

        if not result:
            raise RuntimeError("Training pipeline returned no result.")

        print("\nNew NLP model trained successfully.")

        return True

    except Exception as error:
        print("\nNLP model training failed.")
        print(f"Error: {error}")

        return False


def main():

    project_root = Path(__file__).resolve().parent

    data_dir = project_root / "data" / "raw"

    model_dir = (
        project_root
        / "artifacts"
        / "final_model"
    )

    hash_file = (
        project_root
        / "artifacts"
        / "training_state"
        / "dataset_hash.json"
    )

    ui_path = (
        project_root
        / "src"
        / "prediction_ui.py"
    )

    print("\n" + "=" * 60)
    print("          ARCHFORGE - NLP APPLICATION")
    print("=" * 60)

    if not ui_path.is_file():
        print("\nPrediction UI file not found.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Detect available datasets.
    current_dataset_hash = calculate_dataset_hash(data_dir)

    if current_dataset_hash:

        previous_dataset_hash = load_previous_dataset_hash(hash_file)

        # 2. Train only when the dataset is new or modified.
        if current_dataset_hash != previous_dataset_hash:

            print("\nNew or modified dataset detected.")

            training_successful = train_model(project_root)

            if not training_successful:
                print("\nApplication startup cancelled.")
                return

            # 3. Verify newly generated model package.
            if not verify_model_package(model_dir):
                print("\nModel package verification failed.")
                print("Application startup cancelled.")
                return

            # Save hash only after successful training and packaging.
            save_dataset_hash(
                hash_file,
                current_dataset_hash,
            )

            print("\nDataset training state updated.")

        else:
            print("\nNo dataset changes detected.")
            print("Using the previously trained model.")

    else:
        print("\nNo CSV dataset found inside data/raw/.")

        if not verify_model_package(model_dir):
            print("\nNo existing trained model package found.")
            print("Place a CSV dataset inside data/raw/ and restart.")
            return

        print("Using the existing trained model.")

    # 4. Verify model package before launching UI.
    if not verify_model_package(model_dir):
        print("\nTrained model package not found or incomplete.")
        return

    # 5. Launch prediction UI.
    print("\nStarting ArchForge NLP Streamlit application...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ui_path),
            "--server.headless",
            "true",
        ],
        cwd=str(project_root),
        check=False,
    )


if __name__ == "__main__":
    main()
