# Save this file as 'model_bayesian.py'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# --- CHANGE 1: Import BayesianRidge ---
from sklearn.linear_model import BayesianRidge 
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
# Import the run_preprocessing function from the preprocessing file
from preprocessing import run_preprocessing, major_feature_engineering, create_preprocessor_pipeline

# --- 1. Load and Prepare Data ---

# Run the complete preprocessing pipeline to get the processed data arrays
X_full_train_proc, X_test_proc, y_full_train_log, test_ids, preprocessor = run_preprocessing(
    train_file="train.csv", 
    test_file="test.csv", 
    target_log_transform=True
)

# Create a Train-Validation Split for metric reporting (80/20 split on processed data)
X_sub_train, X_val, y_sub_train_log, y_val_log = train_test_split(
    X_full_train_proc, y_full_train_log, test_size=0.2, random_state=42
)

# --- 2. Define the Bayesian Model ---
# --- CHANGE 2: Initialize BayesianRidge instead of LinearRegression ---
bayesian_model = BayesianRidge()

# --- 3. Train Model on Full Data and Report Metrics ---

print("\n--- Model Training & Evaluation (Bayesian Ridge Regression) ---")

# Train the model on the sub-training split for evaluation
bayesian_model.fit(X_sub_train, y_sub_train_log)

# Calculate metrics (using log-transformed values)
val_log_predictions = bayesian_model.predict(X_val)
val_r2 = r2_score(y_val_log, val_log_predictions)
val_mse = mean_squared_error(y_val_log, val_log_predictions)

print(f"Validation R-squared (Log-transformed): {val_r2:.4f}")
print(f"Validation Mean Squared Error (Log-transformed): {val_mse:.4f}")

# --- 4. Create Final Production Model and Save ---

print("\nFitting final production model on ALL training data...")

# Re-train the bayesian model on the FULL processed data
bayesian_model.fit(X_full_train_proc, y_full_train_log)

# Save the trained model object for the prediction script
# --- CHANGE 3: Update the output filename ---
joblib.dump(bayesian_model, 'fitted_bayesian_model.joblib')

print("Final trained Bayesian Ridge model saved to 'fitted_bayesian_model.joblib'.")

# Note: The fitted_preprocessor.joblib was saved by run_preprocessing().