import lightgbm as lgb
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from lightgbm import early_stopping

def train_lightgbm(trainX, trainY, testX, testY, feature_names, look_back=10):
    """
    Trains a LightGBM model and calculates RMSE on test data,
    handling flattened arrays and grouping feature importances.
    """
    # Reshape the data from 3D to 2D (flatten the time series data)
    trainX_flat = trainX.reshape(trainX.shape[0], -1)  # Flattening to 2D
    testX_flat = testX.reshape(testX.shape[0], -1)  # Flattening to 2D
    
    # Set up LightGBM regressor with default parameters
    model = lgb.LGBMRegressor(verbose=-1)  # Set verbosity to -1 to suppress all messages

    # Define hyperparameters for tuning
    param_grid = {
        'num_leaves': [31, 50],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [100, 200],
        'max_depth': [-1, 10],
    }

    # Early stopping and logging callbacks
    fit_params = {
        "eval_set": [(testX_flat, testY)],
        "eval_metric": "rmse",
        "callbacks": [early_stopping(stopping_rounds=10)]
    }

    # Perform Grid Search for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(trainX_flat, trainY, **fit_params)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Predict on the test set using the best model
    predictions = best_model.predict(testX_flat)

    # Calculate RMSE
    rmse = root_mean_squared_error(testY, predictions)

    # Get feature importances
    flattened_importances = best_model.booster_.feature_importance(importance_type="gain")

    # Map the flattened importances back to the original feature names
    grouped_importances = {name: 0 for name in feature_names}
    for i, importance in enumerate(flattened_importances):
        original_feature_idx = i % len(feature_names)
        original_feature_name = feature_names[original_feature_idx]
        grouped_importances[original_feature_name] += importance

    # Normalize by the number of time steps
    grouped_importances = {k: v / look_back for k, v in grouped_importances.items()}

    sorted_importances = sorted(grouped_importances.items(), key=lambda x: x[1], reverse=True)

    return rmse, best_model, sorted_importances
