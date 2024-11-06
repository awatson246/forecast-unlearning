import numpy as np
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

    # Check if number of features matches the number of feature names
    if len(feature_names) != X_permuted.shape[1]:
        print(f"Warning: Mismatch between number of features ({X_permuted.shape[1]}) and feature_names length ({len(feature_names)})")
        feature_names = feature_names[:X_permuted.shape[1]]  # Adjust to match the data
    
    for idx, feature_name in enumerate(feature_names):
        if idx >= X_permuted.shape[1]:  # Skip if index is out of bounds
            break
        if feature_name == "index":
            continue

        # Permute feature values
        original_values = X_permuted[:, idx].copy()
        np.random.shuffle(X_permuted[:, idx])

        # Make predictions with the permuted feature
        y_pred = model.predict(X_permuted.reshape(X.shape[0], look_back, -1) if model_type == "lstm" else X_permuted)
        
        # Calculate RMSE and store the drop in accuracy
        rmse_permuted = np.sqrt(root_mean_squared_error(y, y_pred))
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