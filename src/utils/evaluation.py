import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb

def evaluate_model(model, trainX, trainY, testX, testY):
    """Evaluates the model and prints metrics."""
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Invert scaling
    trainY, testY = trainY.reshape(-1, 1), testY.reshape(-1, 1)
    scaler = MinMaxScaler().fit(trainY)
    trainPredict, testPredict = scaler.inverse_transform(trainPredict), scaler.inverse_transform(testPredict)
    trainY, testY = scaler.inverse_transform(trainY), scaler.inverse_transform(testY)

    # Calculate and print evaluation metrics
    rmse = np.sqrt(root_mean_squared_error(testY, testPredict))
    r2 = r2_score(testY, testPredict)
    print(f"Model RMSE: {rmse:.4f}")
    print(f"Model R²: {r2:.4f}")

    # Plot Actual vs Predicted
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

def permutation_importance(model, X, y, feature_indices, look_back, model_type, metric=root_mean_squared_error):
    """ Calculate permutation feature importance for time-series data. """
    # Predict baseline error
    y_pred = model.predict(X)
    baseline_error = metric(y, y_pred)
    importance_scores = {}

    for feature_idx in feature_indices:
        # Clone X to permute it without affecting the original
        if model_type == "lstm":
            X_permuted = X.copy()
            for t in range(look_back):
                np.random.shuffle(X_permuted[:, t, feature_idx])  # Shuffle across all time steps
        else:  # For LightGBM or any 2D-input model
            X_permuted = X.copy().values  # Convert to NumPy array
            np.random.shuffle(X_permuted[:, feature_idx])

        # Calculate error with permuted data
        y_pred_permuted = model.predict(X_permuted)
        permuted_error = metric(y, y_pred_permuted)

        # Importance score as increase in error
        importance_scores[feature_idx] = permuted_error - baseline_error

    return importance_scores

def display_feature_importance(model, feature_names):
    importance = model.feature_importance(importance_type='gain')
    for name, score in zip(feature_names, importance):
        print(f"{name}: {score}")

    # Optional: Plot feature importance
    lgb.plot_importance(model, max_num_features=10)
    plt.show()
