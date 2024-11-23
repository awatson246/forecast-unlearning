import numpy as np
from src.models.lstm_model import train_lstm
from src.models.lightgbm_model import train_lightgbm
from src.models.xgboost_model import train_xgboost

def full_retraining(model, trainX, trainY, testX, testY, feature_index, model_type):
    """
    Fully retrains the model after removing the most important feature from the dataset.
    """
    # Remove the most important feature from both train and test data
    retrain_trainX = np.delete(trainX, feature_index, axis=-1)
    retrain_testX = np.delete(testX, feature_index, axis=-1)

    if model_type == "lightgbm":
        retrained_rmse, _ = train_lightgbm(retrain_trainX, trainY, retrain_testX, testY)
    elif model_type == "xgboost":
        retrained_rmse, _ = train_xgboost((retrain_trainX, trainY), (retrain_testX, testY))
    elif model_type == "lstm":
        input_shape = retrain_trainX.shape[1:]
        retrained_rmse, _, _, _ = train_lstm((retrain_trainX, trainY), (retrain_testX, testY), input_shape)

    return retrained_rmse

