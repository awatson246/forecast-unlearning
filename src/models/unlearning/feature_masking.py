import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

def feature_masking(model, data, feature_indices, model_type, true_labels):
    """
    Applies masking to the specified features by replacing them with random noise.
    """
    if data.ndim != 3:
        raise ValueError(f"Input data has incompatible shape: {data.shape}, expected a 3D array.")

    # Copy data and apply masking
    masked_data = data.copy()
    for feature_idx in feature_indices:
        if feature_idx >= data.shape[2]:
            raise ValueError(f"Feature index {feature_idx} is out of bounds for input data with shape {data.shape}.")
        
        # Replace feature values with random noise matching the original feature's distribution
        original_values = data[:, :, feature_idx].flatten()
        noise = np.random.normal(np.mean(original_values), np.std(original_values), size=original_values.shape)
        masked_data[:, :, feature_idx] = noise.reshape(data.shape[0], data.shape[1])
    
    # Flatten for LightGBM, XGBoost, or CatBoost
    masked_data_reshaped = masked_data.reshape(masked_data.shape[0], -1)

    if model_type == "xgboost":
        masked_data_dmatrix = xgb.DMatrix(masked_data_reshaped)
        predictions = model.predict(masked_data_dmatrix)
    elif model_type in ["lightgbm", "catboost"]:
        predictions = model.predict(masked_data_reshaped)

    rmse = np.sqrt(mean_squared_error(true_labels, predictions))
    return rmse, model
