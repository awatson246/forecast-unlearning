import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb

def evaluate_model(model, trainX, trainY, testX, testY):
    """Evaluates the model, prints metrics, and plots actual vs. predicted values."""
    
    # Predict values for train and test data
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Invert scaling for predictions and true values
    scaler = MinMaxScaler().fit(trainY.reshape(-1, 1))
    trainPredict = scaler.inverse_transform(trainPredict.reshape(-1, 1))
    testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))
    trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
    testY = scaler.inverse_transform(testY.reshape(-1, 1))

    # Calculate and display RMSE and R² scores
    rmse = np.sqrt(root_mean_squared_error(testY, testPredict))
    r2 = r2_score(testY, testPredict)
    print(f"Model RMSE: {rmse:.4f}")
    print(f"Model R²: {r2:.4f}")

    # Plot Actual vs Predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(trainY, label='Actual Train Data')
    plt.plot(trainPredict, label='Predicted Train Data')
    plt.plot(np.arange(len(trainY), len(trainY) + len(testY)), testY, label='Actual Test Data')
    plt.plot(np.arange(len(trainY), len(trainY) + len(testY)), testPredict, label='Predicted Test Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.show()

def permutation_importance(model, X, y, feature_names, look_back, model_type):
    """Calculates, ranks, and returns the permutation importance of all features."""
    
    # Ensure X is a NumPy array
    X_permuted = X.copy() if isinstance(X, np.ndarray) else X.values
    feature_importance = {}

    for idx, feature_name in enumerate(feature_names):
        if feature_name in ("index", "year", "month", "day"):
            continue
        
        original_values = X_permuted[:, idx].copy()

        if model_type == "lstm":
            X_permuted = X.copy()
            for t in range(look_back):
                np.random.shuffle(X_permuted[:, t, idx])  # Shuffle across all time steps
        else:  # For LightGBM or any 2D-input model
            X_permuted = X.copy().values  # Convert to NumPy array
            np.random.shuffle(X_permuted[:, idx])

        # Calculate error with permuted data
        y_pred_permuted = model.predict(X_permuted)

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
        print(f"{feature_name}: RMSE with Permutation = {score:.4f}")

    # Save the most important feature for unlearning
    most_important_feature = sorted_importance[0][0]
    print(f"\nMost important feature for unlearning: {most_important_feature}")

    return sorted_importance, most_important_feature


def evaluate_unlearning(model, X, y, unlearned_X, model_type):
    """Evaluates the unlearning process and computes RMSE."""

    # Evaluate initial RMSE
    if model_type == "lightgbm":
        X = X.reshape(X.shape[0], -1)  # Reshape for LightGBM (2D)
    elif model_type == "lstm":
        pass  # Keep X in 3D for LSTM

    print(f"Shape of X: {X.shape}")

    y_pred_initial = model.predict(X)
    initial_rmse = np.sqrt(np.mean((y - y_pred_initial) ** 2))

    # Reshape unlearned_X for LightGBM (2D) and LSTM (3D)
    if model_type == "lightgbm":
        unlearned_X = unlearned_X.reshape(unlearned_X.shape[0], -1)  # Flatten to 2D for LightGBM
    elif model_type == "lstm":
        pass  # Keep unlearned_X in 3D for LSTM

    # Predict with unlearned data
    y_pred_unlearned = model.predict(unlearned_X)
    unlearned_rmse = np.sqrt(np.mean((y - y_pred_unlearned) ** 2))

    return initial_rmse, unlearned_rmse