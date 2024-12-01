from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error

def train_catboost(trainX, trainY, testX, testY, feature_names, look_back=10):
    """
    Trains a CatBoost model and calculates RMSE on the test set, handling grouped feature importance.
    """
    # Reshape the data from 3D to 2D (flatten the time series data)
    trainX_flat = trainX.reshape(trainX.shape[0], -1)
    testX_flat = testX.reshape(testX.shape[0], -1)

    # Set up CatBoost regressor with default parameters
    model = CatBoostRegressor(iterations=500, depth=10, learning_rate=0.05, loss_function="RMSE", verbose=0)

    # Train the model
    model.fit(trainX_flat, trainY, eval_set=(testX_flat, testY), early_stopping_rounds=50)

    # Predict on the test set using the trained model
    predictions = model.predict(testX_flat)

    # Calculate RMSE
    rmse = root_mean_squared_error(testY, predictions)

    # Get raw feature importances
    flattened_importance = model.get_feature_importance()

    # Map the flattened importances back to the original feature names
    grouped_importances = {name: 0 for name in feature_names}
    for i, importance in enumerate(flattened_importance):
        original_feature_idx = i % len(feature_names)
        original_feature_name = feature_names[original_feature_idx]
        grouped_importances[original_feature_name] += importance

    # Normalize by the number of time steps
    grouped_importances = {k: v / look_back for k, v in grouped_importances.items()}

    sorted_importances = sorted(grouped_importances.items(), key=lambda x: x[1], reverse=True)

    return rmse, model, sorted_importances
