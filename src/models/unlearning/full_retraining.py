import numpy as np
from src.models.lstm_model import train_lstm
from src.models.lightgbm_model import train_lightgbm
from src.models.xgboost_model import train_xgboost

def full_retraining(model, trainX, trainY, testX, testY, feature_index, model_type, look_back=10):
    """
    Fully retrains the model after removing the most important feature from the dataset.
    """
    # Check model type and retrain accordingly
    if model_type == "lightgbm":
        # Ensure LightGBM input is 2D
        retrained_rmse, model = train_lightgbm(trainX, trainY, testX, testY)
    elif model_type == "xgboost":
        # Ensure XGBoost input is 2D
        retrained_rmse, model = train_xgboost((trainX, trainY), (testX, testY))
    elif model_type == "lstm":
        # LSTM expects 3D input; no need to reshape
        print(f"Shape of retrain_trainX: {trainX.shape}")
        
        # Pass the data directly if already in 3D format
        if len(trainX.shape) == 3:
            retrain_trainX_lstm = trainX
            retrain_testX_lstm = testX
        else:
            raise ValueError(f"LSTM model requires 3D input; got {trainX.shape} instead")
        
        retrained_rmse, model, _, _ = train_lstm((retrain_trainX_lstm, trainY), (retrain_testX_lstm, testY), retrain_trainX_lstm.shape[1:])
    
    return retrained_rmse, model





