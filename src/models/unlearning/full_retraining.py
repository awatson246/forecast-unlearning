import numpy as np
from src.models.lightgbm_model import train_lightgbm
from src.models.xgboost_model import train_xgboost
from src.models.catboost_model import train_catboost

def full_retraining(model, trainX, trainY, testX, testY, feature_columns, model_type, feature_names):
    """
    Fully retrains the model after removing multiple features.
    Args:
        model: The trained model (LightGBM, XGBoost, etc.)
        trainX, trainY: Training data.
        testX, testY: Test data.
        feature_columns: List of feature column indices to mask.
        model_type: The type of model ('lightgbm', 'xgboost', etc.)
        feature_names: List of feature names.
        
    Returns:
        Retrained RMSE and the retrained model.
    """
    # Create a copy of the data to avoid modifying the original
    retrained_trainX = trainX.copy()
    retrained_testX = testX.copy()

    # Masking multiple columns at once
    for col in feature_columns:
        retrained_trainX[:, :, col] = 0
        retrained_testX[:, :, col] = 0

    # Flatten the data for LightGBM and XGBoost
    retrained_trainX_reshaped = retrained_trainX.reshape(retrained_trainX.shape[0], -1)  # Flatten to 2D
    retrained_testX_reshaped = retrained_testX.reshape(retrained_testX.shape[0], -1)  # Flatten to 2D
    
    # Retrain the model with the masked data
    if model_type == "lightgbm":
        retrained_rmse, model, sorted_importances = train_lightgbm(retrained_trainX_reshaped, trainY, retrained_testX_reshaped, testY, feature_names)
    elif model_type == "xgboost":
        retrained_rmse, model, sorted_importances = train_xgboost((retrained_trainX_reshaped, trainY), (retrained_testX_reshaped, testY), feature_names)
    elif model_type == "catboost":
        retrained_rmse, model, sorted_importances = train_catboost(retrained_trainX_reshaped, trainY, retrained_testX_reshaped, testY, feature_names)
    
    return retrained_rmse, model, sorted_importances



