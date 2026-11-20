from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():

    # Step 1: Load and split data
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Step 2: Transform data
    transformation = DataTransformation()
    train_array, test_array = transformation.initiate_data_transformation(
        train_path, test_path
    )

    # Step 3: Train and evaluate models
    trainer = ModelTrainer()
    best_score = trainer.initiate_model_trainer(
        train_array, test_array
    )

    print("Training pipeline completed successfully!")
    print(f"Best R2 Score: {best_score:.4f}")


if __name__ == "__main__":
    run_training_pipeline()