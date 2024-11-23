import numpy as np
from tensorflow.python.keras.models import clone_model

def fine_tune_after_masking(model, trainX, trainY, masked_trainX, model_type):
    if model_type == "lightgbm":
        # Fine-tune LightGBM
        model.fit(masked_trainX, trainY, verbose=False)
    elif model_type == "xgboost":
        # Fine-tune XGBoost
        dtrain = xgb.DMatrix(data=masked_trainX, label=trainY)
        model.train(dtrain)
    elif model_type == "lstm":
        # Fine-tune LSTM
        model.fit(masked_trainX, trainY, epochs=5, batch_size=32, verbose=0)
    return model