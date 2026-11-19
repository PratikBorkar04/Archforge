
"""
training_pipeline.py

Complete NLP model training workflow.
"""

from pathlib import Path

from src.data_ingestion import NLPDataIngestion
from src.data_validation import NLPDataValidation
from src.text_preprocessing import NLPTextPreprocessing
from src.tokenizer import NLPTokenizer
from src.data_transformation import NLPDataTransformation
from src.model_trainer import NLPModelTrainer
from src.model_pusher import NLPModelPusher


class NLPTrainingPipeline:

    def __init__(self, project_root: Path):

        self.project_root = Path(project_root).resolve()

    def initiate_training_pipeline(self):

        print("\n" + "=" * 60)
        print("       ARCHFORGE - NLP TRAINING PIPELINE")
        print("=" * 60)

        # 1. DATA INGESTION
        print("\n[1/7] Data ingestion...")

        ingestion = NLPDataIngestion(self.project_root)
        ingestion.initiate_data_ingestion()

        # 2. DATA VALIDATION
        print("\n[2/7] Data validation...")

        validation = NLPDataValidation(self.project_root)
        validation.initiate_data_validation()

        # 3. TEXT PREPROCESSING
        print("\n[3/7] NLP preprocessing...")

        preprocessing = NLPTextPreprocessing(
            self.project_root
        )

        (
            processed_train,
            processed_validation,
            processed_test,
            text_columns,
            target_column,
        ) = preprocessing.initiate_text_preprocessing()

        # 4. TOKENIZATION
        print("\n[4/7] Text tokenization...")

        tokenizer = NLPTokenizer(self.project_root)

        (
            tokenized_train,
            tokenized_validation,
            tokenized_test,
        ) = tokenizer.initiate_tokenization()

        # 5. DATA TRANSFORMATION
        print("\n[5/7] TF-IDF transformation...")

        transformation = NLPDataTransformation(
        self.project_root,
        target_column=target_column,        )

        (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test,
            vectorizer,
        ) = transformation.initiate_data_transformation()

        # 6. MODEL TRAINING
        print("\n[6/7] Model training...")

        trainer = NLPModelTrainer(self.project_root)

        (
            best_model,
            best_model_name,
            model_results,
        ) = trainer.initiate_model_training()

        # 7. MODEL PUSHING
        print("\n[7/7] Model packaging...")

        model_pusher = NLPModelPusher(
            project_root=self.project_root,
            target_column=target_column,
            text_columns=text_columns,
        )

        (
            final_model_path,
            final_vectorizer_path,
            metadata_path,
        ) = model_pusher.initiate_model_pusher()

        # SUMMARY
        print("\n" + "=" * 60)
        print("          TRAINING PIPELINE SUMMARY")
        print("=" * 60)

        print(f"\nTarget        : {target_column}")
        print(f"Text columns  : {', '.join(text_columns)}")

        print(f"Training rows : {X_train.shape[0]}")
        print(f"Testing rows  : {X_test.shape[0]}")
        print(f"TF-IDF features: {X_train.shape[1]}")

        print(f"\nBest model: {best_model_name}")

        print("\nModel comparison:")

        for name, scores in model_results.items():

            print(
                f"{name}: "
                f"Accuracy={scores['accuracy']:.4f}, "
                f"F1={scores['f1_score']:.4f}"
            )

        print("\nFinal model package:")
        print(final_model_path)
        print(final_vectorizer_path)
        print(metadata_path)

        print("\nNLP training pipeline completed successfully.")
        print("=" * 60)

        return {
            "best_model": best_model_name,
            "model_results": model_results,
            "target_column": target_column,
            "text_columns": text_columns,
            "model_path": str(final_model_path),
            "vectorizer_path": str(final_vectorizer_path),
            "metadata_path": str(metadata_path),
        }
