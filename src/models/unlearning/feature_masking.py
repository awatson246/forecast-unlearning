import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

def apply_feature_masking(model, data, feature_index, model_type, true_labels):
    # Ensure data has the correct 3D shape and enough features to mask
    if data.ndim != 3:
        raise ValueError(f"Input data has incompatible shape: {data.shape}, expected a 3D array.")
    if data.shape[2] <= feature_index:
        raise ValueError(f"Feature index {feature_index} is out of bounds for input data with shape {data.shape}.")
    
    # Copy data to avoid modifying the original data
    masked_data = data.copy()

    # Masking
    masked_data[:, :, feature_index] = 0  # Example masking logic

    # For XGBoost and LightGBM, convert the masked data into 2D (flatten the 3D array)
    if model_type in ["xgboost", "lightgbm"]:
        masked_data_reshaped = masked_data.reshape(masked_data.shape[0], -1)  # Flatten the 3D array to 2D
        if model_type == "xgboost":
            masked_data_dmatrix = xgb.DMatrix(masked_data_reshaped)
            predictions = model.predict(masked_data_dmatrix)
        elif model_type == "lightgbm":
            predictions = model.predict(masked_data_reshaped)
    else:
        # For LSTM and other model types, just use the regular numpy array (3D input for LSTM)
        predictions = model.predict(masked_data)
    
    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(true_labels, predictions))
    
    return rmse, model
