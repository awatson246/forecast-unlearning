import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import xgboost as xgb

def permutation_importance(model, X, y, feature_names, look_back, model_type):
    """Calculates, ranks, and returns the permutation importance of all features."""
    
    # Ensure X is a NumPy array
    X_permuted = X.copy() if isinstance(X, np.ndarray) else X.values
    feature_importance = {}

    for idx, feature_name in enumerate(feature_names):
        if feature_name in ("index", "year", "month", "day", "Row ID", "Date"):
            continue
        
        # Make sure we don't over-index
        if idx < X_permuted.shape[1]:
            original_values = X_permuted[:, idx].copy()
        else:
            continue

        if model_type == "lstm":
            X_permuted = X.copy()
            for t in range(look_back):
                np.random.shuffle(X_permuted[:, t, idx])  # Shuffle across all time steps
            X_input = X_permuted
        else:  # For LightGBM or any 2D-input model
            if isinstance(X, pd.DataFrame): 
                X_permuted = X.copy().values 
            else: 
                X_permuted = np.copy(X)

            np.random.shuffle(X_permuted[:, idx])
            X_input = X_permuted.reshape(X_permuted.shape[0], -1)

            if model_type == "xgboost":
                X_input = xgb.DMatrix(data=X_input)
        
        # Now make the prediction using the appropriate data type
        y_pred_permuted = model.predict(X_input)

        # Calculate RMSE and store the drop in accuracy
        rmse_permuted = np.sqrt(root_mean_squared_error(y, y_pred_permuted))
        feature_importance[feature_name] = rmse_permuted

        # Restore original feature values
        X_permuted[:, idx] = original_values

    # Sort the features by their importance (higher RMSE means higher importance)
    sorted_importance = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)

    # Print sorted permutation importance results
    print("\nPermutation Importance for All Features:")
    for feature_name, score in sorted_importance:
        print(f"{feature_name}: RMSE with Permutation = {score:.8f}")

    # Save the most important feature for unlearning
    most_important_feature = sorted_importance[0][0]
    print(f"\nMost important feature for unlearning: {most_important_feature}")

    return sorted_importance, most_important_feature


def feature_sensitivity_analysis(model, testX, testY, feature_index, model_type, dataset):
    """
    Evaluate feature sensitivity for a trained model.
    """
    sensitivity_scores = []
    
    if model_type == "lstm":
        # Use the original 3D shape for LSTM
        X_input = testX
    else:
        # Flatten input for non-LSTM models
        X_input = testX.reshape(testX.shape[0], -1)
        if model_type == "xgboost":
            X_input = xgb.DMatrix(data=X_input)

    # Predict using the correct input format
    original_predictions = model.predict(X_input)

    for i in range(testX.shape[-1]):  # Loop through all features
        if model_type == "lstm":
            # Ensure the feature index is valid
            if i >= testX.shape[2]:
                print(f"Skipping index {i} as it exceeds the feature dimension of {testX.shape[2]}")
                continue

            # Perturb the selected feature across all time steps
            perturbed_testX = testX.copy()
            for t in range(testX.shape[1]):
                perturbed_testX[:, t, i] = np.random.permutation(perturbed_testX[:, t, i])

            perturbed_input = perturbed_testX
        else:
            # Flatten input for non-LSTM models and ensure valid feature index
            perturbed_testX = testX.copy() if isinstance(testX, pd.DataFrame) else np.copy(testX)
            if i >= perturbed_testX.shape[1]:
                print(f"Skipping index {i} as it exceeds the feature dimension of {perturbed_testX.shape[1]}")
                continue

            perturbed_testX[:, i] = np.random.permutation(perturbed_testX[:, i])
            perturbed_input = perturbed_testX.reshape(perturbed_testX.shape[0], -1)
            if model_type == "xgboost":
                perturbed_input = xgb.DMatrix(data=perturbed_input)

        # Predict with perturbed data
        perturbed_predictions = model.predict(perturbed_input)

        # Calculate sensitivity as RMSE difference
        sensitivity = root_mean_squared_error(original_predictions, perturbed_predictions)
        sensitivity_scores.append(sensitivity)

        # Sensitivity for masked feature
        if i == feature_index:
            print(f"Sensitivity for masked feature {i}: {sensitivity}")
            masked_sensitivity = sensitivity

    return masked_sensitivity, sensitivity_scores
