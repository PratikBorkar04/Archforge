from pathlib import Path

from src.data_ingestion import NLPDataIngestion
from src.data_validation import NLPDataValidation
from src.text_preprocessing import NLPTextPreprocessing


def main():

    project_root = Path(__file__).resolve().parent

    raw_data_dir = project_root / "data" / "raw"
    artifacts_dir = project_root / "artifacts"

    print("\n" + "=" * 60)
    print("        ARCHFORGE - NLP PIPELINE TEST")
    print("=" * 60)

    print(f"\nRaw data : {raw_data_dir}")
    print(f"Artifacts: {artifacts_dir}")

    # ==================================================
    # CHECK RAW DATA
    # ==================================================

    csv_files = list(
        raw_data_dir.glob("*.csv")
    )

    if not csv_files:

        print(
            "\n❌ No CSV dataset found in data/raw/"
        )

        return

    print("\nDataset found:")

    for file in csv_files:

        print(
            f"  → {file.name}"
        )

    # ==================================================
    # DATA INGESTION
    # ==================================================

    print(
        "\n" + "-" * 60
    )

    print(
        "Running NLP data ingestion..."
    )

    print(
        "-" * 60
    )

    try:

        ingestion = NLPDataIngestion(
            project_root=project_root
        )

        (
            train_data,
            validation_data,
            test_data,
        ) = ingestion.initiate_data_ingestion()

        print(
            "\n✅ NLP data ingestion completed."
        )

    except Exception as exc:

        print(
            "\n❌ NLP data ingestion failed."
        )

        print(
            f"\nError: {exc}"
        )

        raise

    # ==================================================
    # DATA VALIDATION
    # ==================================================

    print(
        "\n" + "-" * 60
    )

    print(
        "Running NLP data validation..."
    )

    print(
        "-" * 60
    )

    try:

        validation = NLPDataValidation(
            project_root=project_root
        )

        (
            validated_train,
            validated_validation,
            validated_test,
            validation_reports,
        ) = validation.initiate_data_validation()

        print(
            "\n✅ NLP data validation completed."
        )

    except Exception as exc:

        print(
            "\n❌ NLP data validation failed."
        )

        print(
            f"\nError: {exc}"
        )

        raise

    # ==================================================
    # TEXT PREPROCESSING
    # ==================================================

    print(
        "\n" + "-" * 60
    )

    print(
        "Running NLP text preprocessing..."
    )

    print(
        "-" * 60
    )

    try:

        preprocessing = NLPTextPreprocessing(
            project_root=project_root
        )

        (
            processed_train,
            processed_validation,
            processed_test,
        ) = (
            preprocessing.initiate_text_preprocessing(
                text_column="review"
            )
        )

        print(
            "\n✅ NLP text preprocessing completed."
        )

    except Exception as exc:

        print(
            "\n❌ NLP text preprocessing failed."
        )

        print(
            f"\nError: {exc}"
        )

        raise

    # ==================================================
    # PIPELINE SUMMARY
    # ==================================================

    validation_count = (
        len(processed_validation)
        if processed_validation is not None
        else "Not provided"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "        NLP PIPELINE TEST SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"\nTraining samples   : "
        f"{len(processed_train)}"
    )

    print(
        f"Validation samples : "
        f"{validation_count}"
    )

    print(
        f"Testing samples    : "
        f"{len(processed_test)}"
    )

    print(
        f"\nValidation reports : "
        f"{len(validation_reports)}"
    )

    print(
        "\nGenerated artifacts:"
    )

    if artifacts_dir.exists():

        generated_files = [
            file
            for file in artifacts_dir.rglob("*")
            if file.is_file()
        ]

        for file in generated_files:

            print(
                f"  → "
                f"{file.relative_to(artifacts_dir)}"
            )

    print(
        "\n" + "=" * 60
    )

    print(
        "✅ NLP ingestion + validation + "
        "preprocessing completed."
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()