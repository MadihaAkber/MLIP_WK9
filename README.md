
# Student Performance Prediction | End-to-End Machine Learning

An end-to-end machine learning project that predicts students' mathematics scores using academic and demographic features.

Developed as part of Week 9 of the MSc Applied Artificial Intelligence programme at the University of Westminster.

## Project Overview

This project demonstrates the complete machine learning workflow, from data ingestion and preprocessing to model training, evaluation and deployment through a Flask web application.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost and CatBoost
- Flask
- HTML and CSS
- Git and GitHub

## Machine Learning Models

Six regression algorithms were evaluated:

| Model | R² Score |
|---|---:|
| Linear Regression | 0.8804 |
| Ridge Regression | 0.8806 |
| Lasso Regression | 0.8564 |
| Random Forest | 0.8488 |
| XGBoost | 0.8231 |
| CatBoost | 0.8511 |

**Best-performing model: Ridge Regression (R² = 0.8806)**

These results are from the project's held-out test split. Model selection was also performed on that split, so a separate validation process would be needed for a rigorous final performance estimate.

## Project Features

- Automated data ingestion and train-test splitting
- Numerical and categorical feature preprocessing
- Missing-value imputation, scaling and one-hot encoding
- Training and comparison of multiple regression algorithms
- Model serialization using Joblib
- Reusable prediction pipeline
- Flask-based web application for interactive predictions
- Custom logging and exception handling

## Running the Application

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m src.pipeline.train_pipeline
```

Start the Flask application:

```bash
python app.py
```

Open the application on port 5001.

## Example Prediction

For an example student with a reading score of 75 and writing score of 80, alongside the selected demographic and educational inputs, the model predicted a mathematics score of approximately **71.12**.

## Future Improvements

- Use separate training, validation and test sets
- Add cross-validation and hyperparameter tuning
- Improve form validation and error handling
- Deploy the application to a permanent hosting platform

## Author

Madiha Fatima Agha

MBBS | MSc Applied Artificial Intelligence, University of Westminster

GitHub: https://github.com/MadihaAkber
