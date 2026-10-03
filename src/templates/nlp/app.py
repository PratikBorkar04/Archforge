from pathlib import Path

from src.data_ingestion import NLPDataIngestion
from src.data_validation import NLPDataValidation
from src.text_preprocessing import NLPTextPreprocessing


def main():

    project_root = Path(__file__).resolve().parent

    raw_data_dir = (
        project_root / "data" / "raw"
    )

    artifacts_dir = (
        project_root / "artifacts"
    )

    print("\n" + "=" * 60)
    print("        ARCHFORGE - NLP PIPELINE TEST")
    print("=" * 60)

    # ==================================================
    # CHECK DATA
    # ==================================================

    csv_files = list(
        raw_data_dir.glob("*.csv")
    )

    if not csv_files:

        print(
            "\n❌ No CSV dataset found in data/raw/"
        )

        return

    print(
        f"\nDataset: {csv_files[0].name}"
    )

    # ==================================================
    # DATA INGESTION
    # ==================================================

    print(
        "\n[1/3] Data ingestion..."
    )

    ingestion = NLPDataIngestion(
        project_root=project_root
    )

    (
        train_data,
        validation_data,
        test_data,
    ) = ingestion.initiate_data_ingestion()

    print(
        "✓ Ingestion completed."
    )

    # ==================================================
    # DATA VALIDATION
    # ==================================================

    print(
        "\n[2/3] Data validation..."
    )

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
        "✓ Validation completed."
    )

    # ==================================================
    # NLP COLUMN ANALYSIS + PREPROCESSING
    # ==================================================

    print(
        "\n[3/3] NLP column analysis + preprocessing..."
    )

    preprocessing = NLPTextPreprocessing(
        project_root=project_root
    )

    (
        processed_train,
        processed_validation,
        processed_test,
        text_columns,
        target_column,
    ) = (
        preprocessing
        .initiate_text_preprocessing()
    )

    print(
        "\n✓ NLP preprocessing completed."
    )

    # ==================================================
    # FINAL SUMMARY
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
        "              PIPELINE SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"\nTarget       : {target_column}"
    )

    print(
        f"Text columns : "
        f"{', '.join(text_columns)}"
    )

    print(
        f"\nTrain        : "
        f"{len(processed_train)}"
    )

    print(
        f"Validation   : "
        f"{validation_count}"
    )

    print(
        f"Test         : "
        f"{len(processed_test)}"
    )

    print(
        "\n✓ Current NLP pipeline completed."
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()