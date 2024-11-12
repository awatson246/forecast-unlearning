import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error

def train_xgboost(train, test):
    """ Train an XGBoost model and evaluate RMSE on the test set.    """

    trainX, trainY = train
    testX, testY = test
    
    # Convert data to DMatrix format (required by XGBoost)
    train_data = xgb.DMatrix(trainX, label=trainY)
    test_data = xgb.DMatrix(testX, label=testY)

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
    rmse = np.sqrt(root_mean_squared_error(testY, y_pred))
    print(f"XGBoost Model RMSE: {rmse}")

    return rmse, model
