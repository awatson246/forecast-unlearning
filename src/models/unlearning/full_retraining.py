import numpy as np
from src.models.lightgbm_model import train_lightgbm
from src.models.xgboost_model import train_xgboost
from src.models.catboost_model import train_catboost

def full_retraining(model, trainX, trainY, testX, testY, feature_columns, model_type, feature_names):
    """
    Fully retrains the model after masking multiple features with random noise.
    
    Args:
        model: The trained model (LightGBM, XGBoost, etc.)
        trainX, trainY: Training data.
        testX, testY: Test data.
        feature_columns: List of feature column indices to mask.
        model_type: The type of model ('lightgbm', 'xgboost', etc.)
        feature_names: List of feature names.
        
    Returns:
        Retrained RMSE, the retrained model, and feature importance.
    """
    # Create a copy of the data to avoid modifying the original
    retrained_trainX = trainX.copy()
    retrained_testX = testX.copy()

    # Masking multiple columns with random noise
    for col in feature_columns:
        if col >= retrained_trainX.shape[2]:
            raise ValueError(f"Feature index {col} is out of bounds for input data with shape {retrained_trainX.shape}.")

        # Generate random noise matching the original feature's distribution
        train_col_values = retrained_trainX[:, :, col].flatten()
        test_col_values = retrained_testX[:, :, col].flatten()

        train_noise = np.random.normal(np.mean(train_col_values), np.std(train_col_values), size=train_col_values.shape)
        test_noise = np.random.normal(np.mean(test_col_values), np.std(test_col_values), size=test_col_values.shape)

        # Apply the noise
        retrained_trainX[:, :, col] = train_noise.reshape(retrained_trainX.shape[0], retrained_trainX.shape[1])
        retrained_testX[:, :, col] = test_noise.reshape(retrained_testX.shape[0], retrained_testX.shape[1])

    # Flatten the data for LightGBM, XGBoost, and CatBoost
    retrained_trainX_reshaped = retrained_trainX.reshape(retrained_trainX.shape[0], -1)  # Flatten to 2D
    retrained_testX_reshaped = retrained_testX.reshape(retrained_testX.shape[0], -1)  # Flatten to 2D
    
    # Retrain the model with the masked data
    if model_type == "lightgbm":
        retrained_rmse, model, sorted_importances = train_lightgbm(retrained_trainX_reshaped, trainY, retrained_testX_reshaped, testY, feature_names)
    elif model_type == "xgboost":
        retrained_rmse, model, sorted_importances = train_xgboost((retrained_trainX_reshaped, trainY), (retrained_testX_reshaped, testY), feature_names)
    elif model_type == "catboost":
        retrained_rmse, model, sorted_importances = train_catboost(retrained_trainX_reshaped, trainY, retrained_testX_reshaped, testY, feature_names)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return retrained_rmse, model, sorted_importances
