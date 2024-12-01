import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

def feature_masking(model, data, feature_columns, model_type, true_labels):
    """
    Masks multiple features at once and computes RMSE after masking.
    Args:
        model: The trained model (LightGBM, XGBoost, etc.)
        data: Input data (3D numpy array).
        feature_columns: List of feature column indices to mask.
        model_type: The type of model ('lightgbm', 'xgboost', etc.)
        true_labels: True labels for RMSE calculation.
        
    Returns:
        RMSE after masking
        Model used for prediction
    """
    # Ensure data has the correct shape
    if data.ndim != 3:
        raise ValueError(f"Input data has incompatible shape: {data.shape}, expected a 3D array.")
    
    # Copy data to avoid modifying the original data
    masked_data = data.copy()
    
    # Masking multiple columns (ensure col is an integer)
    for col in feature_columns:
        if isinstance(col, int):  # Ensure col is an integer
            masked_data[:, :, col] = 0  # Mask all specified columns
        else:
            raise ValueError(f"Column index {col} is not an integer.")
    
    # Flatten the data for LightGBM and XGBoost
    masked_data_reshaped = masked_data.reshape(masked_data.shape[0], -1)  # Flatten the 3D array to 2D
    if model_type == "xgboost":
        masked_data_dmatrix = xgb.DMatrix(masked_data_reshaped)
        predictions = model.predict(masked_data_dmatrix)
    else:
        predictions = model.predict(masked_data_reshaped)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(true_labels, predictions))
    return rmse, model
