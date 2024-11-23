import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error

def train_xgboost(train_data, test_data):
    """Trains an XGBoost model and calculates RMSE on the test set."""
    trainX, trainY = train_data
    testX, testY = test_data

    # Flatten the input data to 2D
    trainX_flat = trainX.reshape(trainX.shape[0], -1)
    testX_flat = testX.reshape(testX.shape[0], -1)

    # Prepare DMatrix objects for training and testing
    train_data = xgb.DMatrix(trainX_flat, label=trainY)
    test_data = xgb.DMatrix(testX_flat, label=testY)

    # Define XGBoost parameters
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,  # Suppress logging output
    }

    # Train the model with early stopping
    evals = [(train_data, "train"), (test_data, "test")]
    model = xgb.train(params, train_data, num_boost_round=200, evals=evals, early_stopping_rounds=10)

    # Predict on the test set
    predictions = model.predict(test_data)

    # Calculate RMSE
    rmse = root_mean_squared_error(testY, predictions)

    return rmse, model
