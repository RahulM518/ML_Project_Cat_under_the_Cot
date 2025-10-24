import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import xgboost as xgb # You may need to install this: pip install xgboost

# --- Imports for Model Evaluation ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("Starting the XGBoost-only workflow (Simplified)...")
# --- 1. Load Data ---
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: train.csv or test.csv not found.")
    print("Please make sure the files are in the same directory as the script.")
    exit()

# --- 2. Preprocessing (Simplified) ---

# Store IDs for final submission
train_id = train_df['Id']
test_id = test_df['Id']

# Drop Id column as it's not a feature
train_df = train_df.drop('Id', axis=1)
test_df = test_df.drop('Id', axis=1)

# Analyze and Transform Target Variable (HotelValue)
y_train_log = np.log1p(train_df['HotelValue'])
train_df = train_df.drop('HotelValue', axis=1)

# Combine Train and Test Data for Consistent Preprocessing
ntrain = train_df.shape[0]
ntest = test_df.shape[0]
all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
print(f"Combined data shape: {all_data.shape}")

# --- (NEW) Remove Unnecessary Columns ---
# These columns are >90% empty and are not worth keeping
cols_to_drop = ['ServiceLaneType', 'PoolQuality', 'BoundaryFence', 'ExtraFacility']
all_data = all_data.drop(cols_to_drop, axis=1)
print(f"Data shape after dropping unnecessary columns: {all_data.shape}")


# Handle Remaining Missing Values (Imputation)
print("Handling missing values...")
numerical_cols = all_data.select_dtypes(include=[np.number]).columns
categorical_cols = all_data.select_dtypes(include=['object']).columns

# Fill categorical NaNs with 'None' (e.g., for ParkingType, BasementHeight)
for col in categorical_cols:
    all_data[col] = all_data[col].fillna('None')

# Fill numerical NaNs with the median (e.g., for RoadAccessLength, FacadeArea)
for col in numerical_cols:
    all_data[col] = all_data[col].fillna(all_data[col].median())

# One-Hot Encoding
print("Applying one-hot encoding...")
all_data_processed = pd.get_dummies(all_data, drop_first=True)

# Re-split into Train and Test
X_train_processed = all_data_processed[:ntrain]
X_test_processed = all_data_processed[ntrain:]

print(f"Processed training features shape: {X_train_processed.shape}")
print(f"Processed test features shape: {X_test_processed.shape}")


# --- 3. (PART 1) XGBoost Model Evaluation ---
print("\n--- Starting Model Evaluation ---")

# Split the 1200 rows into a 80%/20% (train/validation) split
X_train_eval, X_val, y_train_eval_log, y_val_log = train_test_split(
    X_train_processed, y_train_log, test_size=0.2, random_state=42
)

# Scale this new split data
# (XGBoost is not sensitive to scaling, but it's not bad practice)
eval_scaler = StandardScaler()
X_train_eval_scaled = eval_scaler.fit_transform(X_train_eval)
X_val_scaled = eval_scaler.transform(X_val)

# Train XGBoost model for evaluation
print("Training XGBoost model for evaluation...")
eval_xgb = xgb.XGBRegressor(
    objective='reg:squarederror', n_estimators=5000, learning_rate=0.03,
    max_depth=6, subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=-1
)

# Use the validation set for early stopping
eval_xgb.fit(X_train_eval_scaled, y_train_eval_log,
             early_stopping_rounds=50,
             eval_set=[(X_val_scaled, y_val_log)],
             verbose=False)

# Predict on the 20% validation set
val_preds_xgb_log = eval_xgb.predict(X_val_scaled)

# Convert predictions and "correct answers" back to dollars
y_val_dollars = np.expm1(y_val_log)
val_preds_xgb_dollars = np.expm1(val_preds_xgb_log)

# Calculate and print metrics
print("\n--- XGBoost Evaluation Results (Simplified Model) ---")
xgb_r2 = r2_score(y_val_dollars, val_preds_xgb_dollars)
xgb_mae = mean_absolute_error(y_val_dollars, val_preds_xgb_dollars)
xgb_rmse = np.sqrt(mean_squared_error(y_val_dollars, val_preds_xgb_dollars))
print(f"  R-squared (R2): {xgb_r2:.4f}")
print(f"  Mean Absolute Error (MAE): ${xgb_mae:,.2f}")
print(f"  Root Mean Squared Error (RMSE): ${xgb_rmse:,.2f}")

print("\n--- Evaluation Complete. Proceeding to create final submission file... ---")


# --- 4. (PART 2) Final XGBoost Model for Submission ---

# Now, we use a *new* scaler and fit it on ALL 1200 training rows
print("\nScaling all training data for final submission...")
final_scaler = StandardScaler()
X_train_scaled = final_scaler.fit_transform(X_train_processed) # 1200 rows
X_test_scaled = final_scaler.transform(X_test_processed) # 260 rows

# We re-train the model on 100% of the training data
print("Training final XGBoost model on all data...")
final_xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    # We can use the 'best_ntree_limit' from the evaluation model to avoid overfitting
    n_estimators=2500,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.6,
    random_state=42,
    n_jobs=-1
)
# We fit on *all* the training data
final_xgb_model.fit(X_train_scaled, y_train_log)

# Predict on the actual test set (260 rows)
xgb_preds_log = final_xgb_model.predict(X_test_scaled)

# Inverse Transform Predictions
xgb_preds = np.expm1(xgb_preds_log)

# Create Submission File for XGBoost
xgb_submission = pd.DataFrame({'Id': test_id, 'HotelValue': xgb_preds})
xgb_submission.to_csv('xgboost_submission.csv', index=False)
print("Created 'xgboost_submission.csv'")

print("\n--- All Steps Complete! ---")