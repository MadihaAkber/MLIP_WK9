import os
import sys
import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts", "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Random Forest": RandomForestRegressor(
                    random_state=42
                ),
                "XGBoost": XGBRegressor(
                    random_state=42
                ),
                "CatBoost": CatBoostRegressor(
                    verbose=False,
                    random_seed=42
                )
            }

            best_score = float("-inf")
            best_model = None
            best_model_name = None

            for name, model in models.items():
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                r2 = r2_score(y_test, predictions)
                rmse = np.sqrt(
                    mean_squared_error(y_test, predictions)
                )

                print(f"{name}: R2={r2:.4f}, RMSE={rmse:.4f}")

                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = name

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(
                f"Best model: {best_model_name}, R2: {best_score}"
            )

            print(f"Best Model: {best_model_name}")
            print(f"Best R2 Score: {best_score:.4f}")

            return best_score

        except Exception as e:
            raise CustomException(e, sys)