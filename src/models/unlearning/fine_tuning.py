import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

def fine_tuning(model, trainX, trainY, testX, testY, feature_columns, model_type):
    """
    Fine-tunes the model after masking multiple features at once.
    Args:
        model: The trained model (LightGBM, XGBoost, etc.)
        trainX, trainY: Training data.
        testX, testY: Test data.
        feature_columns: List of feature column indices to mask.
        model_type: The type of model ('lightgbm', 'xgboost', etc.)
        
    Returns:
        Fine-tuned RMSE and the fine-tuned model.
    """
    # Create a copy of the data to avoid modifying the original
    masked_trainX = trainX.copy()
    masked_testX = testX.copy()

    # Masking multiple columns at once
    for col in feature_columns:
        masked_trainX[:, :, col] = 0
        masked_testX[:, :, col] = 0
    
    # Flatten the data for LightGBM and XGBoost
    masked_trainX_reshaped = masked_trainX.reshape(masked_trainX.shape[0], -1)  # Flatten to 2D
    masked_testX_reshaped = masked_testX.reshape(masked_testX.shape[0], -1)  # Flatten to 2D
    
    # Fine-tune the model with the masked data
    if model_type in ["lightgbm", "catboost"]:
        fine_tuned_predictions = model.predict(masked_testX_reshaped)
    elif model_type == "xgboost":
        fine_tuned_predictions = model.predict(xgb.DMatrix(masked_testX_reshaped))
    
    # Calculate RMSE for the fine-tuned model
    fine_tuned_rmse = np.sqrt(mean_squared_error(testY, fine_tuned_predictions))
    
    return fine_tuned_rmse, model
