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

def permutation_importance(model, X, y, feature_indices, look_back, model_type):
    """Calculates and returns the permutation importance of specified features."""
    
    # Ensure X is a NumPy array
    X_permuted = X.copy() if isinstance(X, np.ndarray) else X.values
    feature_importance = {}

    for idx in feature_indices:
        # Permute feature values
        original_values = X_permuted[:, idx].copy()
        np.random.shuffle(X_permuted[:, idx])

        # Make predictions with the permuted feature
        y_pred = model.predict(X_permuted.reshape(X.shape[0], look_back, -1) if model_type == "lstm" else X_permuted)
        
        # Calculate RMSE and store the drop in accuracy
        rmse_permuted = np.sqrt(root_mean_squared_error(y, y_pred))
        feature_importance[idx] = rmse_permuted

        # Restore original feature values
        X_permuted[:, idx] = original_values

    # Print permutation importance results
    print("\nPermutation Importance for Selected Features:")
    for idx, score in feature_importance.items():
        print(f"Feature {idx}: RMSE with Permutation = {score:.4f}")

    return feature_importance

def display_feature_importance(model, feature_names):
    """Displays feature importance from the LightGBM model based on gain and plots the top features."""
    
    importance = model.feature_importance(importance_type='gain')
    print("\nFeature Importances:")
    for name, score in zip(feature_names, importance):
        print(f"{name}: {score}")

    # Optional: Plot feature importance
    lgb.plot_importance(model, max_num_features=10, importance_type='gain', title='Top Feature Importances')
    plt.show()
