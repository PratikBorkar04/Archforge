from pathlib import Path

from src.data_ingestion import NLPDataIngestion


def main():

    project_root = Path(__file__).resolve().parent

    raw_data_dir = (
        project_root
        / "data"
        / "raw"
    )

    artifacts_dir = (
        project_root
        / "artifacts"
    )

    print("\n" + "=" * 60)
    print("        ARCHFORGE - NLP DATA INGESTION TEST")
    print("=" * 60)

    print(
        f"\nRaw data : {raw_data_dir}"
    )

    print(
        f"Artifacts: {artifacts_dir}"
    )

    # --------------------------------------------------
    # Check dataset
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Run ingestion
    # --------------------------------------------------

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

        result = (
            ingestion.initiate_data_ingestion()
        )

        print(
            "\n✅ NLP data ingestion completed."
        )

        print("\nResult:")

        print(result)

    except Exception as exc:

        print(
            "\n❌ NLP data ingestion failed."
        )

        print(
            f"\nError: {exc}"
        )

        raise

    # --------------------------------------------------
    # Show generated artifacts
    # --------------------------------------------------

    print(
        "\n" + "-" * 60
    )

    print(
        "Generated artifacts"
    )

    print(
        "-" * 60
    )

    if artifacts_dir.exists():

        generated_files = [
            file
            for file in artifacts_dir.rglob("*")
            if file.is_file()
        ]

        if generated_files:

            for file in generated_files:

                print(
                    f"  → "
                    f"{file.relative_to(artifacts_dir)}"
                )

        else:

            print(
                "  No files generated."
            )

    else:

        print(
            "  Artifacts directory does not exist."
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "DONE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()