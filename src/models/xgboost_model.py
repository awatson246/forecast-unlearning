import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error

def train_xgboost(train, test, target_column):
    """
    Train an XGBoost model and evaluate RMSE on the test set.
    
    Args:
        train: Training data.
        test: Test data.
        target_column: Column name of the target variable.
    
    Returns:
        model: Trained XGBoost model.
    """
    # Separate features and target
    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]
    
    # Convert data to DMatrix format (required by XGBoost)
    train_data = xgb.DMatrix(X_train, label=y_train)
    test_data = xgb.DMatrix(X_test, label=y_test)

    # Set XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }

    # Train the model with early stopping
    evals = [(test_data, "eval")]
    model = xgb.train(params, train_data, num_boost_round=100, early_stopping_rounds=10, evals=evals, verbose_eval=10)

    # Predict and calculate RMSE on test set
    y_pred = model.predict(test_data)
    rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))
    print(f"XGBoost Model RMSE: {rmse}")

    return model
