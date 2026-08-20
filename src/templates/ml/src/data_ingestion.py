"""
Data Ingestion
--------------

Automatically discovers and loads the user's dataset from:

    data/raw/

For the first ArchForge ML version, one CSV dataset is expected.

The component:
    1. Finds the CSV dataset automatically.
    2. Loads the dataset.
    3. Separates features (X) and target (y).
    4. Creates training and testing datasets.
    5. Saves the split datasets inside artifacts/.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:
    """
    Automatically handles dataset discovery and train/test splitting.

    Parameters
    ----------
    project_root : Path
        Root directory of the generated ML project.

    test_size : float, optional
        Percentage of data reserved for testing.
        Default is 0.20 (20%).

    random_state : int, optional
        Ensures reproducible train/test splitting.
        Default is 42.
    """

    def __init__(
        self,
        project_root: Path,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:

        self.project_root = Path(project_root).resolve()

        self.raw_data_dir = self.project_root / "data" / "raw"
        self.artifacts_dir = self.project_root / "artifacts"

        self.test_size = test_size
        self.random_state = random_state

    def _find_dataset(self) -> Path:
        """
        Automatically find the CSV dataset.

        Returns
        -------
        Path
            Path to the discovered dataset.

        Raises
        ------
        FileNotFoundError
            If no CSV dataset is found.

        ValueError
            If multiple CSV datasets are found.
        """

        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.raw_data_dir}"
            )

        csv_files = list(self.raw_data_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                "\nNo CSV dataset found.\n"
                f"Please place your dataset inside:\n"
                f"{self.raw_data_dir}"
            )

        if len(csv_files) > 1:
            file_names = ", ".join(
                file.name for file in csv_files
            )

            raise ValueError(
                "\nMultiple CSV datasets found:\n"
                f"{file_names}\n\n"
                "Please keep only one dataset inside "
                "data/raw/ for automatic training."
            )

        return csv_files[0]

    def _load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        """
        Load the discovered CSV dataset.

        Parameters
        ----------
        dataset_path : Path
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """

        try:
            data = pd.read_csv(dataset_path)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to read dataset: {dataset_path}"
            ) from exc

        if data.empty:
            raise ValueError(
                "The dataset is empty. "
                "Please provide a dataset containing data."
            )

        return data

    def initiate_data_ingestion(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Discover, load and split the dataset automatically.

        The last column is treated as the target variable.

        Returns
        -------
        tuple
            train_data, test_data
        """

        print("\n[1/6] Data Ingestion")

        # ----------------------------------------------------------
        # Step 1: Find dataset automatically
        # ----------------------------------------------------------

        dataset_path = self._find_dataset()

        print(f"Dataset found: {dataset_path.name}")

        # ----------------------------------------------------------
        # Step 2: Load dataset
        # ----------------------------------------------------------

        data = self._load_dataset(dataset_path)

        print(
            f"Dataset shape: "
            f"{data.shape[0]} rows × {data.shape[1]} columns"
        )

        # ----------------------------------------------------------
        # Step 3: Validate basic dataset structure
        # ----------------------------------------------------------

        if data.shape[1] < 2:
            raise ValueError(
                "Dataset must contain at least two columns: "
                "features and target."
            )

        # ----------------------------------------------------------
        # Step 4: Treat the last column as target
        # ----------------------------------------------------------

        target_column = data.columns[-1]

        print(f"Target column detected: {target_column}")

        # ----------------------------------------------------------
        # Step 5: Split training and testing data
        # ----------------------------------------------------------

        train_data, test_data = train_test_split(
            data,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # ----------------------------------------------------------
        # Step 6: Save datasets for reproducibility
        # ----------------------------------------------------------

        self.artifacts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_path = self.artifacts_dir / "train.csv"
        test_path = self.artifacts_dir / "test.csv"

        train_data.to_csv(
            train_path,
            index=False,
        )

        test_data.to_csv(
            test_path,
            index=False,
        )

        print(f"Training data saved: {train_path}")
        print(f"Testing data saved: {test_path}")

        return train_data, test_data