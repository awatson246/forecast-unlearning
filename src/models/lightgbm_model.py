import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import math

def train_lightgbm(trainX, trainY, testX, testY):
    """Trains a LightGBM model and calculates RMSE on test data, with hyperparameter tuning."""
    
    # Set up LightGBM regressor with default parameters
    model = lgb.LGBMRegressor()

    # Define hyperparameters for tuning
    param_grid = {
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [-1, 10, 20],
        'min_data_in_leaf': [20, 50, 100]
    }

    # Perform Grid Search for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    grid_search.fit(trainX, trainY)
    
    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Predict on the test set using the best model
    predictions = best_model.predict(testX)
    
    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(testY, predictions))
    
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"RMSE on Test Set: {rmse:.4f}")
    
    return rmse, best_model
