"""
model_trainer.py

Model training component for ArchForge.

Responsibilities
----------------
1. Receive models selected by ModelSelector.
2. Instantiate selected models.
3. Train selected models.
4. Record training status and time.
5. Return trained models.

This module does NOT:
- Select models.
- Decide dataset safety.
- Evaluate models.
- Save models.

Optional model libraries:
- XGBoost
- CatBoost

If these libraries are not installed, their models are skipped
safely when selected.
"""

from pathlib import Path
from typing import Dict, Any, List
import time


# ================================================================
# Optional Gradient Boosting Libraries
# ================================================================

try:
    from xgboost import (
        XGBRegressor,
        XGBClassifier,
    )

    XGBOOST_AVAILABLE = True

except ImportError:
    XGBRegressor = None
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


try:
    from catboost import (
        CatBoostRegressor,
        CatBoostClassifier,
    )

    CATBOOST_AVAILABLE = True

except ImportError:
    CatBoostRegressor = None
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False


# ================================================================
# Scikit-learn Models
# ================================================================

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier,
)

from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier,
)

from sklearn.svm import (
    SVR,
    SVC,
)

from sklearn.naive_bayes import (
    GaussianNB,
)


class ModelTrainer:
    """
    Train models selected by ModelSelector.
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

    # ============================================================
    # Main Training Method
    # ============================================================

    def initiate_model_training(
        self,
        train_data: Any,
        selected_models: List[str],
        target_column: str,
        validation_data: Any = None,
    ) -> Dict[str, Any]:
        """
        Train models selected by ModelSelector.

        Parameters
        ----------
        train_data:
            Training data.

        selected_models:
            List of model names returned by ModelSelector.

        target_column:
            Target column name.

        validation_data:
            Optional validation data.

            This parameter is accepted for pipeline compatibility,
            but validation is NOT performed in this component.

            Evaluation belongs to model_evaluation.py.

        Returns
        -------
        Dict[str, Any]
            Trained models and training metadata.
        """

        if train_data is None:
            raise ValueError(
                "Training data cannot be None."
            )

        if not selected_models:
            raise ValueError(
                "No models were selected."
            )

        if not target_column:
            raise ValueError(
                "Target column is required."
            )

        # --------------------------------------------------------
        # Prepare training data
        # --------------------------------------------------------

        X_train, y_train = (
            self._prepare_training_data(
                train_data=train_data,
                target_column=target_column,
            )
        )

        feature_count = (
            X_train.shape[1]
            if hasattr(
                X_train,
                "shape",
            )
            else 0
        )

        sample_count = len(X_train)

        # --------------------------------------------------------
        # Compact header
        # --------------------------------------------------------

        print(
            f"Training {len(selected_models)} model(s) "
            f"on {sample_count} samples..."
        )

        # --------------------------------------------------------
        # Containers
        # --------------------------------------------------------

        trained_models: Dict[
            str, Any
        ] = {}

        training_metadata: Dict[
            str, Dict[str, Any]
        ] = {}

        # --------------------------------------------------------
        # Train selected models
        # --------------------------------------------------------

        for model_name in selected_models:

            start_time = time.perf_counter()

            try:

                model = self._create_model(
                    model_name
                )

                model.fit(
                    X_train,
                    y_train,
                )

                elapsed_time = (
                    time.perf_counter()
                    - start_time
                )

                trained_models[
                    model_name
                ] = model

                training_metadata[
                    model_name
                ] = {
                    "status": "SUCCESS",
                    "training_time_seconds": round(
                        elapsed_time,
                        4,
                    ),
                    "samples": int(
                        sample_count
                    ),
                    "features": int(
                        feature_count
                    ),
                }

                print(
                    f"  ✓ {model_name} "
                    f"({elapsed_time:.2f}s)"
                )

            except Exception as exc:

                elapsed_time = (
                    time.perf_counter()
                    - start_time
                )

                training_metadata[
                    model_name
                ] = {
                    "status": "FAILED",
                    "training_time_seconds": round(
                        elapsed_time,
                        4,
                    ),
                    "samples": int(
                        sample_count
                    ),
                    "features": int(
                        feature_count
                    ),
                    "error": str(exc),
                }

                print(
                    f"  ✗ {model_name}: {exc}"
                )

        # --------------------------------------------------------
        # At least one model must succeed
        # --------------------------------------------------------

        if not trained_models:

            raise RuntimeError(
                "All selected models failed "
                "during training."
            )

        # --------------------------------------------------------
        # Summary
        # --------------------------------------------------------

        successful_models = [
            name
            for name, metadata
            in training_metadata.items()
            if metadata["status"] == "SUCCESS"
        ]

        failed_models = [
            name
            for name, metadata
            in training_metadata.items()
            if metadata["status"] == "FAILED"
        ]

        print(
            f"Training completed: "
            f"{len(successful_models)} succeeded, "
            f"{len(failed_models)} failed."
        )

        return {
            "trained_models":
                trained_models,

            "training_metadata":
                training_metadata,

            "successful_models":
                successful_models,

            "failed_models":
                failed_models,

            "training_samples":
                int(sample_count),

            "training_features":
                int(feature_count),
        }

    # ============================================================
    # Prepare Training Data
    # ============================================================

    @staticmethod
    def _prepare_training_data(
        train_data: Any,
        target_column: str,
    ):
        """
        Prepare X_train and y_train.

        Supported formats:

        1. DataFrame

        2. Tuple/list:
           (X_train, y_train)

        3. Dictionary:
           {
               "X_train": ...,
               "y_train": ...
           }

        4. Dictionary:
           {
               "train_features": ...,
               "train_target": ...
           }
        """

        # --------------------------------------------------------
        # Dictionary
        # --------------------------------------------------------

        if isinstance(
            train_data,
            dict,
        ):

            if (
                "X_train" in train_data
                and
                "y_train" in train_data
            ):

                return (
                    train_data["X_train"],
                    train_data["y_train"],
                )

            if (
                "train_features" in train_data
                and
                "train_target" in train_data
            ):

                return (
                    train_data[
                        "train_features"
                    ],
                    train_data[
                        "train_target"
                    ],
                )

            if "train_data" in train_data:

                train_data = (
                    train_data[
                        "train_data"
                    ]
                )

        # --------------------------------------------------------
        # Tuple / List
        # --------------------------------------------------------

        if isinstance(
            train_data,
            (tuple, list),
        ):

            if len(train_data) >= 2:

                return (
                    train_data[0],
                    train_data[1],
                )

            raise ValueError(
                "Training data tuple/list "
                "must contain X_train and y_train."
            )

        # --------------------------------------------------------
        # DataFrame
        # --------------------------------------------------------

        if hasattr(
            train_data,
            "columns",
        ):

            if (
                target_column
                not in train_data.columns
            ):

                raise ValueError(
                    f"Target column "
                    f"'{target_column}' "
                    f"not found in training data."
                )

            X_train = train_data.drop(
                columns=[
                    target_column
                ]
            )

            y_train = train_data[
                target_column
            ]

            return (
                X_train,
                y_train,
            )

        raise TypeError(
            "Unsupported training data format."
        )

    # ============================================================
    # Model Factory
    # ============================================================

    @staticmethod
    def _create_model(
        model_name: str,
    ) -> Any:
        """
        Create a model instance from its name.

        XGBoost and CatBoost are optional dependencies.
        """

        models = {

            # ----------------------------------------------------
            # Regression
            # ----------------------------------------------------

            "LinearRegression":
                LinearRegression(),

            "Ridge":
                Ridge(),

            "Lasso":
                Lasso(),

            "ElasticNet":
                ElasticNet(),

            "DecisionTreeRegressor":
                DecisionTreeRegressor(
                    random_state=42
                ),

            "RandomForestRegressor":
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),

            "GradientBoostingRegressor":
                GradientBoostingRegressor(
                    random_state=42
                ),

            "AdaBoostRegressor":
                AdaBoostRegressor(
                    random_state=42
                ),

            "KNeighborsRegressor":
                KNeighborsRegressor(),

            "SVR":
                SVR(),

            # ----------------------------------------------------
            # Classification
            # ----------------------------------------------------

            "LogisticRegression":
                LogisticRegression(
                    max_iter=1000
                ),

            "DecisionTreeClassifier":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "RandomForestClassifier":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                ),

            "GradientBoostingClassifier":
                GradientBoostingClassifier(
                    random_state=42
                ),

            "AdaBoostClassifier":
                AdaBoostClassifier(
                    random_state=42
                ),

            "KNeighborsClassifier":
                KNeighborsClassifier(),

            "SVC":
                SVC(),

            "GaussianNB":
                GaussianNB(),
        }

        # --------------------------------------------------------
        # XGBoost
        # --------------------------------------------------------

        if XGBOOST_AVAILABLE:

            models.update({

                "XGBRegressor":
                    XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                        objective="reg:squarederror",
                    ),

                "XGBClassifier":
                    XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                        eval_metric="logloss",
                    ),

            })

        # --------------------------------------------------------
        # CatBoost
        # --------------------------------------------------------

        if CATBOOST_AVAILABLE:

            models.update({

                "CatBoostRegressor":
                    CatBoostRegressor(
                        iterations=200,
                        learning_rate=0.05,
                        depth=6,
                        loss_function="RMSE",
                        random_seed=42,
                        verbose=False,
                        allow_writing_files=False,
                    ),

                "CatBoostClassifier":
                    CatBoostClassifier(
                        iterations=200,
                        learning_rate=0.05,
                        depth=6,
                        loss_function="Logloss",
                        random_seed=42,
                        verbose=False,
                        allow_writing_files=False,
                    ),

            })

        # --------------------------------------------------------
        # Validate model
        # --------------------------------------------------------

        if model_name not in models:

            if model_name in (
                "XGBRegressor",
                "XGBClassifier",
            ) and not XGBOOST_AVAILABLE:

                raise ImportError(
                    f"{model_name} requires "
                    "the 'xgboost' package. "
                    "Install it using: "
                    "pip install xgboost"
                )

            if model_name in (
                "CatBoostRegressor",
                "CatBoostClassifier",
            ) and not CATBOOST_AVAILABLE:

                raise ImportError(
                    f"{model_name} requires "
                    "the 'catboost' package. "
                    "Install it using: "
                    "pip install catboost"
                )

            raise ValueError(
                f"Unsupported model: "
                f"{model_name}"
            )

        return models[
            model_name
        ]

    # ============================================================
    # Public Model Factory Helper
    # ============================================================

    @staticmethod
    def get_model(
        model_name: str,
    ) -> Any:
        """
        Public helper for creating a model.
        """

        return ModelTrainer._create_model(
            model_name
        )
